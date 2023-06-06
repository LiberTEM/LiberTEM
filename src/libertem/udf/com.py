from typing import NamedTuple, Optional, Tuple, Union
from collections import namedtuple

import numpy as np
from sparseconverter import CUPY_BACKENDS

from libertem import masks
from libertem.corrections import coordinates
from libertem.udf.masks import ApplyMasksEngine
from libertem.common.container import MaskContainer
from libertem.common.math import prod
from libertem.udf.base import UDF


COMParams = namedtuple(
    'COMParams',
    ('cy', 'cx', 'r', 'ri', 'scan_rotation', 'flip_y'),
    defaults=(None, None, None, None, None, None)
)


def com_masks_factory(detector_y, detector_x, cy, cx, r):
    def disk_mask():
        return masks.circular(
            centerX=cx, centerY=cy,
            imageSizeX=detector_x,
            imageSizeY=detector_y,
            radius=r,
        )

    return [
        disk_mask,
        lambda: masks.gradient_y(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * disk_mask(),
        lambda: masks.gradient_x(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * disk_mask(),
    ]


def com_masks_generic(detector_y, detector_x, base_mask_factory):
    """
    Create a CoM mask stack with a generic selection mask factory

    Parameters
    ----------
    detector_y : int
        The detector height
    detector_x : int
        The detector width
    base_mask_factory : () -> np.array
        A factory function for creating the selection mask

    Returns
    -------
    List[Function]
        The mask stack as a list of factory functions
    """
    return [
        base_mask_factory,
        lambda: masks.gradient_y(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * base_mask_factory(),
        lambda: masks.gradient_x(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * base_mask_factory(),
    ]


def center_shifts(img_sum, img_y, img_x, ref_y, ref_x):
    x_centers = np.divide(img_x, img_sum, where=img_sum != 0)
    y_centers = np.divide(img_y, img_sum, where=img_sum != 0)
    x_centers[img_sum == 0] = ref_x
    y_centers[img_sum == 0] = ref_y
    x_centers -= ref_x
    y_centers -= ref_y
    return (y_centers, x_centers)


def apply_correction(y_centers, x_centers, scan_rotation, flip_y, forward=True):
    shape = y_centers.shape
    if flip_y:
        transform = coordinates.flip_y()
    else:
        transform = coordinates.identity()
    # Transformations are applied right to left
    transform = coordinates.rotate_deg(scan_rotation) @ transform
    y_centers = y_centers.reshape(-1)
    x_centers = x_centers.reshape(-1)
    if not forward:
        transform = np.linalg.inv(transform)
    y_transformed, x_transformed = transform @ (y_centers, x_centers)
    y_transformed = y_transformed.reshape(shape)
    x_transformed = x_transformed.reshape(shape)
    return (y_transformed, x_transformed)


def divergence(y_centers, x_centers):
    return np.gradient(y_centers, axis=0) + np.gradient(x_centers, axis=1)


def curl_2d(y_centers, x_centers):
    # https://en.wikipedia.org/wiki/Curl_(mathematics)#Usage
    # DFy/dx - dFx/dy
    # axis 0 is y, axis 1 is x
    return np.gradient(y_centers, axis=1) - np.gradient(x_centers, axis=0)


def magnitude(y_centers, x_centers):
    return np.sqrt(y_centers**2 + x_centers**2)


def coordinate_check(y_centers, x_centers, roi=None):
    '''
    Calculate the RMS curl as a function of :code:`scan_rotation` and :code:`flip_y`.

    The curl for a purely electrostatic field is zero. That means
    the correct settings for :code:`scan_rotation` and :code:`flip_y` should
    minimize the RMS curl for atomic resolution STEM of non-magnetic specimens
    along a high-symmetry zone axis.

    Parameters
    ----------
    y_centers, x_centers : numpy.ndarray
        2D arrays with y and x component of the center of mass shift for each
        scan position, as returned by :meth:`center_shifts` or
        :meth:`apply_correction`
    roi : Optional[numpy.ndarray]
        Selector for values to consider in the statistics, compatible
        with indexing an array with the shape of y_centers and x_centers.
        By default, everything except the last row and last column are used
        since these contain artefacts.

    Returns
    -------
    (straight, flipped)
        Root mean square of the curl as a function of :code:`scan_rotation` from
        0 to 359 degrees in steps of one, with :code:`flip_y=False` (straight)
        and code:`flip_y=True` (flipped).
    '''
    straight = np.zeros(360)
    flipped = np.zeros(360)
    if roi is None:
        # The last row and column contain artifacts
        roi = (slice(0, -1), slice(0, -1))
    for angle in range(360):
        for flip_y in (True, False):
            y_transformed, x_transformed = apply_correction(
                y_centers, x_centers, scan_rotation=angle, flip_y=flip_y
            )
            curl = curl_2d(y_transformed, x_transformed)
            result = np.sqrt(np.mean(curl[roi]**2))
            if flip_y:
                flipped[angle] = result
            else:
                straight[angle] = result
    return (straight, flipped)


class GuessResult(NamedTuple):
    scan_rotation: int
    flip_y: bool
    cy: float
    cx: float


def guess_corrections(
    y_centers: np.ndarray,
    x_centers: np.ndarray,
    roi: Optional[Union[np.ndarray, Tuple[slice, ...]]] = None,
) -> GuessResult:
    '''
    Guess corrections for center shift, :code:`scan_rotation` and :code:`flip_y` from CoM data

    This function can generate a CoM parameter guess for atomic resolution 4D STEM data
    by using the following assumptions:

    * The field is purely electrostatic, i.e. the RMS curl should be minimized
    * There is no net field over the field of view and no descan error,
      i.e. the mean deflection is zero.
    * Atomic resolution STEM means that the divergence will be negative at atom columns
      and consequently the histogram of divergence will have a stronger tail towards
      negative values than towards positive values.

    If any corrections were applied when generating the input data, please note that the corrections
    should be applied relative to these previous value. In particular, the
    center corrections returned by this function have to be back-transformed to the uncorrected
    coordinate system, for example with :code:`apply_correction(..., forward=False)`

    Parameters
    ----------
    y_centers, x_centers : numpy.ndarray
        2D arrays with y and x component of the center of mass shift for each
        scan position, as returned by :meth:`center_shifts` or
        :meth:`apply_correction`
    roi : Optional[numpy.ndarray]
        Selector for values to consider in the statistics, compatible
        with indexing an array with the shape of y_centers and x_centers.
        By default, everything except the last row and last column are used
        since these contain artefacts.


    Returns
    -------
    GuessResult : relative to current values
    '''
    if roi is None:
        # The last row and column contain artefacts
        roi = (slice(0, -1), slice(0, -1))
    straight, flipped = coordinate_check(y_centers, x_centers, roi=roi)
    # The one with lower minima is the correct one
    flip_y = bool(np.min(flipped) < np.min(straight))
    if flip_y:
        angle = np.argmin(flipped)
    else:
        angle = np.argmin(straight)

    corrected_y, corrected_x = apply_correction(
        y_centers, x_centers, scan_rotation=angle, flip_y=flip_y
    )
    # There are two equivalent angles that minimize RMS curl since a 180°
    # rotation inverts the coordinates and just flips the sign for divergence
    # and curl. To distinguish the two, the distribution of the divergence is
    # analyzed. With negative electrons and positive nuclei, the beam is
    # deflected towards the nuclei and the divergence is negative there. Since
    # the beam is deflected most strongly near the nuclei, the histogram should
    # have more values at the negative end of the range than at the positive
    # end.
    div = divergence(corrected_y, corrected_x)[roi]
    all_range = np.maximum(-np.min(div), np.max(div))
    hist, bins = np.histogram(div, range=(-all_range, all_range), bins=5)
    polarity_off = np.sum(hist[:1]) < np.sum(hist[-1:])
    if polarity_off:
        angle += 180
    if angle > 180:
        angle -= 360
    return GuessResult(
        scan_rotation=int(angle),
        flip_y=flip_y,
        cy=np.mean(y_centers[roi]),
        cx=np.mean(x_centers[roi])
    )


class COMUDF(UDF):
    def __init__(self, com_params: COMParams = COMParams()):
        super().__init__(com_params=com_params)

    def get_backends(self):
        return self.BACKEND_ALL

    def get_result_buffers(self):
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            'raw_mask_result': self.buffer(
                kind='nav', dtype=dtype, extra_shape=(3, ), where='device', use='private'
            ),
            'raw_shifts': self.buffer(
                kind='nav', dtype=dtype, extra_shape=(2, ), use='result_only'
            ),
            'field': self.buffer(
                kind='nav', dtype=dtype, extra_shape=(2, ), use='result_only'
            ),
            'magnitude': self.buffer(
                kind='nav', dtype=dtype, use='result_only'
            ),
            'divergence': self.buffer(
                kind='nav', dtype=dtype, use='result_only'
            ),
            'curl': self.buffer(
                kind='nav', dtype=dtype, use='result_only'
            ),
        }

    def get_params(self) -> COMParams:
        sig_shape = tuple(self.meta.dataset_shape.sig)
        cy = self.params.com_params.cy
        if cy is None:
            cy = sig_shape[0] // 2

        cx = self.params.com_params.cx
        if cx is None:
            cx = sig_shape[1] // 2

        r = self.params.com_params.r
        if r is None:
            r = np.inf

        ri = self.params.com_params.ri

        scan_rotation = self.params.com_params.scan_rotation
        if scan_rotation is None:
            scan_rotation = 0

        flip_y = self.params.com_params.flip_y
        if flip_y is None:
            flip_y = False

        return COMParams(
            cy=cy, cx=cx, r=r, ri=ri, scan_rotation=scan_rotation, flip_y=flip_y,
        )

    def get_task_data(self):
        sig_shape = tuple(self.meta.dataset_shape.sig)
        com_params = self.get_params()
        if len(sig_shape) != 2:
            raise ValueError('COMUDF only works with 2D sig shape.')
        if len(self.meta.dataset_shape.nav) != 2:
            raise ValueError('COMUDF only works with 2D nav shape.')

        if com_params.ri is None:
            mask_factory = com_masks_factory(
                detector_y=sig_shape[0],
                detector_x=sig_shape[1],
                cx=com_params.cx,
                cy=com_params.cy,
                r=com_params.r,
            )
        else:
            mask_factory = com_masks_generic(
                detector_y=sig_shape[0],
                detector_x=sig_shape[1],
                base_mask_factory=lambda: masks.ring(
                    imageSizeY=sig_shape[0],
                    imageSizeX=sig_shape[1],
                    centerY=com_params.cy,
                    centerX=com_params.cx,
                    radius=com_params.r,
                    radius_inner=com_params.ri,
                )
            )
        if self.meta.array_backend in CUPY_BACKENDS:
            backend = self.BACKEND_CUPY
        else:
            backend = self.BACKEND_NUMPY

        masks_container = MaskContainer(
            mask_factories=mask_factory, dtype=np.float32, use_sparse=False,
            count=3, backend=backend
        )
        return {
            'com_params': com_params,
            'engine': ApplyMasksEngine(masks=masks_container, meta=self.meta, use_torch=True)
        }

    def process_tile(self, tile):
        raw_result = self.task_data.engine.process_tile(tile)
        self.results.raw_mask_result[:] += self.forbuf(
            raw_result, self.results.raw_mask_result
        )

    def get_field_results(self, field_y, field_x):
        '''
        To be overwritten in subclasses, such as for iCoM results
        in https://github.com/LiberTEM/LiberTEM-iCoM
        '''
        mag = magnitude(
            y_centers=field_y, x_centers=field_x
        )
        div = divergence(y_centers=field_y, x_centers=field_x)
        curl = curl_2d(y_centers=field_y, x_centers=field_x)
        return {
            'magnitude': mag,
            'divergence': div,
            'curl': curl,
        }

    def get_results(self):
        com_params = self.get_params()
        raw_mask_result = self.results.get_buffer('raw_mask_result')
        raw_shifts = center_shifts(
            img_sum=raw_mask_result.data[..., 0],
            img_y=raw_mask_result.data[..., 1],
            img_x=raw_mask_result.data[..., 2],
            ref_y=com_params.cy,
            ref_x=com_params.cx,
        )
        field = apply_correction(
            y_centers=raw_shifts[0],
            x_centers=raw_shifts[1],
            scan_rotation=com_params.scan_rotation,
            flip_y=com_params.flip_y,
        )
        roi = self.meta.roi
        field_y = field[0]
        field_x = field[1]

        raw_shifts = np.moveaxis(np.array(raw_shifts), 0, -1)
        field = np.moveaxis(np.array(field), 0, -1)

        nav_size = prod(self.meta.dataset_shape.nav)

        results = {
            'raw_shifts': raw_shifts,
            'field': field,
        }

        results.update(self.get_field_results(field_y=field_y, field_x=field_x))

        if self.meta.roi is not None:
            for key in results:
                results[key] = results[key][roi]
        else:
            for key in results:
                results[key] = results[key].reshape((nav_size, -1))
        return results
