import math
import logging
import warnings
from typing import TYPE_CHECKING, Optional, Union
from collections.abc import Sequence
from typing_extensions import Literal

import numpy as np

from libertem.common.exceptions import UDFException
from libertem.io.corrections import CorrectionSet
from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.common.udf import UDFProtocol, UDFMethod

if TYPE_CHECKING:
    from numpy import typing as nt
    from libertem.io.dataset.base import DataSet, Partition

log = logging.getLogger(__name__)

TilingIntent = Union[Literal["partition"], Literal["frame"], Literal["tile"]]


class TilingScheme:
    def __init__(
        self, slices: list[Slice],
        tileshape: Shape, dataset_shape: Shape, intent: Optional[TilingIntent] = None, debug=None
    ):
        self._slices = slices
        self._tileshape = tileshape
        self._dataset_shape = dataset_shape
        self._debug = debug
        self._intent = intent

        if tileshape.nav.dims > 1:
            raise ValueError("tileshape should have flat navigation dimensions")

    def adjust_for_partition(self, partition: "Partition") -> "TilingScheme":
        """
        If the intent is per-partition processing, the tiling scheme must match the
        partition shape exactly. If there is a mismatch, this method returns a
        new scheme that matches the partition.

        Parameters
        ----------
        partition
            The Partition we want to adjust the tiling scheme to.

        Returns
        -------
        TilingScheme
            The adjusted tiling scheme, or this one, if it matches exactly
        """
        partition_size = partition.slice.shape.nav.size
        if partition_size != self.depth and self.intent == "partition":
            # adjust depth to match partition size exactly:
            new_shape = Shape(
                (partition_size,) + tuple(self._tileshape.sig),
                sig_dims=self._tileshape.sig.dims
            )
            return TilingScheme(
                slices=self._slices,
                tileshape=new_shape,
                dataset_shape=self._dataset_shape,
                intent=self._intent,
                debug=self._debug,
            )
        return self

    @classmethod
    def make_for_shape(
        cls,
        tileshape: Shape,
        dataset_shape: Shape,
        intent: Optional[TilingIntent] = None,
        debug=None,
    ) -> "TilingScheme":
        """
        Make a TilingScheme from `tileshape` and `dataset_shape`.

        Note that both in signal and navigation direction there are border
        effects, i.e. if the depth doesn't evenly divide the number of
        frames in the partition (simplified, ROI also applies...), or if
        the signal dimensions of `tileshape` doesn't evenly divide the signal
        dimensions of the `dataset_shape`.

        Parameters
        ----------
        tileshape
            Uniform shape of all tiles. Should have flat navigation axis
            (meaning tileshape.nav.dims == 1) and be contiguous in signal
            dimensions.

        dataset_shape
            Shape of the whole data set. Only the signal part is used.

        intent
            The intent of this scheme (whole partitions, frames or tiles)
            Needs to be set for correct per-partition tiling!
        """
        # FIXME: validate navigation part of the tileshape to be contiguous
        # (i.e. a shape like (1, 1, ..., 1, X1, ..., XN))
        # where X1 is <= the dataset shape at that index, and X2, ..., XN are
        # equal to the dataset shape at that index

        sig_slice = Slice(
            origin=tuple([0] * dataset_shape.sig.dims),
            shape=dataset_shape.sig
        )
        subslices = list(sig_slice.subslices(tileshape.sig))
        return cls(
            slices=subslices,
            tileshape=tileshape,
            dataset_shape=dataset_shape,
            debug=debug,
            intent=intent,
        )

    def __getitem__(self, idx: int) -> Slice:
        return self._slices[idx]

    def __len__(self):
        return len(self._slices)

    def __repr__(self):
        unique_shapes = list({tuple(slice_.shape) for slice_ in self._slices})
        return "<TilingScheme (depth=%d) shapes=%r len=%d>" % (
            self.depth, unique_shapes, len(self._slices),
        )

    @property
    def intent(self) -> Optional[TilingIntent]:
        return self._intent

    @property
    def slices(self):
        """
        signal-only slices for all possible positions
        """
        return list(enumerate(self._slices))

    @property
    def slices_array(self):
        """
        Returns the slices from the schema as a numpy ndarray
        `a` of shape `(n, 2, sig_dims)` with:
        `a[i, 0]` are origins for slice `i`
        `a[i, 1]` are shapes for slice `i`
        """
        sig_dims = self._tileshape.sig.dims
        slices = np.zeros((len(self), 2, sig_dims), dtype=np.int64)
        for idx, slice_ in self.slices:
            slices[idx] = (tuple(slice_.origin), tuple(slice_.shape))
        return slices

    @property
    def shape(self):
        """
        tileshape. note that some border tiles can be smaller!
        """
        return self._tileshape

    @property
    def dataset_shape(self):
        return self._dataset_shape

    @property
    def depth(self):
        return self._tileshape.nav[0]


class Negotiator:
    """
    Tile shape negotiator. The main functionality is in `get_scheme`,
    which, given a `udf`, `dataset` and `read_dtype` will generate
    a `TilingScheme` that is compatible with both the `UDF` and the
    `DataSet`, possibly even optimal.
    """

    def validate(
        self,
        shape: tuple[int, ...],
        ds_sig_shape: tuple[int, ...],
        size: int,
        io_max_size: int,
        itemsize: int,
        base_shape: tuple[int, ...],
        corrections: Optional[CorrectionSet],
    ):
        sig_shape = shape[1:]
        # we need some wiggle room with the size, because there may be a harder
        # lower size value for some cases (for example HDF5, which overrides
        # some of the sizing negotiation we are doing here)
        if any(s > ps for s, ps in zip(sig_shape, ds_sig_shape)):
            raise ValueError("generated tileshape does not fit the partition")
        size_px = max(size, io_max_size) // itemsize
        if prod(shape) > size_px:
            message = "shape %r (%d) does not fit into size %d" % (
                shape, prod(shape), size_px
            )
            # The shape might be exceeded if dead pixel correction didn't find a
            # valid tiling scheme. In that case it falls back to by-frame processing.
            if (
                corrections is not None
                and corrections.get_excluded_pixels() is not None
                and shape[0] == 1
            ):
                warnings.warn(message)
            else:
                raise ValueError(message)
        for dim in range(len(base_shape)):
            # Tile shape always has one nav dim
            # Allow only base shape for nav
            # Allow full frames for sig
            if ((shape[dim] % base_shape[dim] != 0)
                    and (dim == 0 or (dim > 0 and shape[dim] != ds_sig_shape[dim - 1]))):
                raise ValueError(
                    f"The tileshape {shape} is incompatible with base "
                    f"shape {base_shape} and dataset shape {ds_sig_shape} in dimension {dim}."
                )

    def get_scheme(
            self,
            udfs: Sequence[UDFProtocol],
            dataset,
            read_dtype: "nt.DTypeLike",
            approx_partition_shape: Shape,
            roi: Optional[np.ndarray] = None,
            corrections: Optional[CorrectionSet] = None,
    ) -> TilingScheme:
        """
        Generate a :class:`TilingScheme` instance that is
        compatible with both the given `udf` and the
        :class:~`libertem.io.dataset.base.DataSet`.

        Parameters
        ----------

        udfs : Sequence[UDFProtocol]
            The concrete UDFs to optimize the tiling scheme for.
            Depending on the method (tile, frame, partition)
            and preferred total input size and depth.

        dataset : DataSet
            The DataSet instance we generate the scheme for.

        read_dtype
            The dtype in which the data will be fed into the UDF

        approx_partition_shape
            The approximate partition shape that is likely to be used

        roi : np.ndarray
            Region of interest

        corrections : CorrectionSet
            Correction set to consider in negotiation
        """
        itemsize = np.dtype(read_dtype).itemsize

        # FIXME: let the UDF define upper bound for signal size (lower bound, too?)
        # (signal buffers should fit into the L2 cache)
        # try not to waste page faults:
        min_sig_size = dataset.get_min_sig_size()
        ds_sig_shape = dataset.shape.sig

        # This already takes corrections into account through a different pathway
        need_decode = dataset.need_decode(roi=roi, read_dtype=read_dtype, corrections=corrections)

        io_max_size = self._get_io_max_size(dataset, approx_partition_shape, itemsize, need_decode)

        depths = [
            self._get_min_depth(udf, approx_partition_shape)
            for udf in udfs
        ]
        depth = max(depths)  # take the largest min-depth
        base_shape = self._get_base_shape(udfs, dataset, approx_partition_shape, roi)

        intent = self._get_intent(udfs)

        sizes = [
            self._get_size(
                io_max_size, udf, itemsize, approx_partition_shape, base_shape,
            )
            for udf in udfs
        ]
        if intent == "partition":
            size = max(sizes)  # by partition wants to be big, ...
        else:
            size = min(sizes)
        size_px = size // itemsize

        if corrections is not None and corrections.have_corrections():
            # The correction has to make sure that there are no excluded pixels
            # at tile boundaries
            base_shape = corrections.adjust_tileshape(
                tile_shape=base_shape,
                sig_shape=tuple(ds_sig_shape),
                base_shape=base_shape,
            )

        # first, scale `base_shape` up to contain at least `min_sig_size` items:
        min_factors = self._get_scale_factors(
            base_shape,
            containing_shape=ds_sig_shape,
            size=min_sig_size,
        )

        min_base_shape = self._scale_base_shape(base_shape, min_factors)

        # considering the min size, calculate the max depth:
        max_depth = max(1, size_px // prod(min_base_shape))
        if depth > max_depth:
            depth = max_depth

        full_base_shape = (1,) + tuple(base_shape)
        min_factors = (depth,) + tuple(min_factors)

        containing_shape = approx_partition_shape

        factors = self._get_scale_factors(
            full_base_shape,
            containing_shape=containing_shape,
            size=size_px,
            min_factors=min_factors,
        )
        tileshape = self._scale_base_shape(full_base_shape, factors)
        tileshape_orig = tileshape

        # the dataset has a "veto" on the tileshape:
        # FIXME: this veto may break if the base shape was adjusted
        # above, and we need to be careful not to break corrections with this,
        # and also fulfill requests of per-frame reading
        log.debug("tileshape before adjustment: %r", (tileshape,))
        tileshape = tuple(dataset.adjust_tileshape(tileshape, roi=roi))
        log.debug("tileshape after adjustment: %r", (tileshape,))

        # if the veto generated a tileshape that is smaller than the full base shape,
        # we need to re-adjust the full_base_shape
        if tileshape_orig != tileshape:  # make sure we don't change too eagerly:
            if tileshape[0] < full_base_shape[0]:
                full_base_shape = (tileshape[0], *full_base_shape[1:])
            has_pixel_corr = (
                corrections is not None
                and corrections.get_excluded_pixels() is not None
            )
            for (orig, new, sig) in zip(tileshape_orig[1:], tileshape[1:], ds_sig_shape):
                if new != orig and new != sig:
                    # Otherwise we may generate incorrect correction results
                    err_str = (
                        "dataset.adjust_tileshape() can only accept tile sig shape or switch to"
                        "full frames if dead pixel patching is active. "
                        f"Got original tile shape {tileshape_orig}, new tileshape {tileshape} "
                        f"and dataset sig shape {ds_sig_shape}"
                    )
                    if has_pixel_corr:
                        raise ValueError(err_str)
                    else:
                        warnings.warn(err_str)

        self.validate(
            tileshape, ds_sig_shape, size, io_max_size, itemsize, full_base_shape, corrections,
        )
        return TilingScheme.make_for_shape(
            tileshape=Shape(tileshape, sig_dims=ds_sig_shape.dims),
            dataset_shape=dataset.shape,
            intent=intent,
            debug={
                "min_factors": min_factors,
                "factors": factors,
                "tileshape": tileshape,
                "size": size,
                "size_px": size_px,
                "full_base_shape": full_base_shape,
                "need_decode": need_decode,
                "depth": depth,
            }
        )

    def _get_io_max_size(self, dataset, approx_partition_shape, itemsize, need_decode):
        if need_decode:
            io_max_size = dataset.get_max_io_size()
            if io_max_size is None:
                io_max_size = 2**20
        else:
            io_max_size = itemsize * prod(approx_partition_shape)
        return io_max_size

    def _get_scale_factors(self, shape, containing_shape, size, min_factors=None):
        """
        Generate scaling factors to scale `shape` up to `size` elements,
        while being constrained to `containing_shape`.
        """
        log.debug(
            "_get_scale_factors in: shape=%r, containing_shape=%r, size=%r, min_factors=%r",
            shape, containing_shape, size, min_factors
        )
        assert len(shape) == len(containing_shape)
        if min_factors is None:
            factors = [1] * len(shape)
        else:
            factors = list(min_factors)
        max_factors = tuple(
            cs // s
            for s, cs in zip(shape, containing_shape)
        )
        prelim_shape = self._scale_base_shape(shape, factors)
        rest = size / prod(prelim_shape)
        if rest < 1:
            rest = 1
        for idx in range(len(shape)):
            max_factor = max_factors[idx]
            factor = int(math.floor(rest * factors[idx]))
            if factor < factors[idx]:
                factor = factors[idx]
            if factor > max_factor:
                factor = max_factor
            factors[idx] = factor
            prelim_shape = self._scale_base_shape(shape, factors)
            rest = max(1, math.floor(size / prod(prelim_shape)))
        log.debug(
            "_get_scale_factors out: %r",
            factors,
        )
        return factors

    def _scale_base_shape(self, base_shape, factors):
        assert len(factors) == len(base_shape)
        return tuple(
            f * bs
            for f, bs in zip(factors, base_shape)
        )

    def _get_default_size(self):
        # FIXME: adjust size to L3 // number of workers per node
        return 1*2**20

    def _get_udf_size_pref(self, udf: UDFProtocol):
        udf_prefs = udf.get_tiling_preferences()
        size = udf_prefs.get("total_size", np.inf)
        if size is UDFProtocol.TILE_SIZE_BEST_FIT:
            size = self._get_default_size()
        return size

    def _get_intent(self, udfs: Sequence[UDFProtocol]) -> TilingIntent:
        udf_methods = tuple(udf.get_method() for udf in udfs)
        if any(m not in tuple(UDFMethod) for m in udf_methods):
            raise UDFException('A UDF declared an invalid processing method')
        if UDFMethod.PARTITION in udf_methods:
            return "partition"
        elif UDFMethod.FRAME in udf_methods:
            return "frame"
        elif UDFMethod.TILE in udf_methods:
            return "tile"
        else:
            raise ValueError('No recognized UDF method, empty udfs arg?')

    def _get_size(
            self, io_max_size, udf: UDFProtocol, itemsize,
            approx_partition_shape: Shape, base_shape):
        """
        Calculate the maximum tile size in bytes
        """
        udf_method = udf.get_method()
        partition_size = itemsize * prod(tuple(approx_partition_shape))
        partition_size_sig = itemsize * prod(tuple(approx_partition_shape.sig))
        if udf_method == UDFMethod.FRAME:
            size = max(self._get_default_size(), partition_size_sig)
        elif udf_method == UDFMethod.PARTITION:
            size = partition_size
        elif udf_method == UDFMethod.TILE:
            # start with the UDF size preference:
            size = self._get_udf_size_pref(udf)

            # constrain to maximum read size
            size = min(size, io_max_size)

            # if the base_shape is larger than the current maximum size,
            # we need to increase the size:
            base_size = itemsize * prod(base_shape)
            size = max(base_size, size)
        else:  # pragma: no cover
            # Should never be reached, this is checked earlier in UDFRunner
            raise UDFException(f'UDF.get_method() returned unrecognized method: {udf_method}')
        return size

    def _get_base_shape(
        self,
        udfs: Sequence["UDFProtocol"],
        dataset: "DataSet",
        approx_partition_shape: Shape,
        roi: Optional[np.ndarray],
    ):
        methods = [
            udf.get_method()
            for udf in udfs
        ]
        if any(m in (UDFMethod.FRAME, UDFMethod.PARTITION) for m in methods):
            base_shape = approx_partition_shape.sig
        else:
            # only by tile:
            base_shape = Shape(
                dataset.get_base_shape(roi=roi),
                sig_dims=approx_partition_shape.sig.dims
            ).sig
        return base_shape

    def _get_udf_depth_pref(self, udf: "UDFProtocol", approx_partition_shape: Shape) -> int:
        udf_prefs = udf.get_tiling_preferences()
        depth = udf_prefs.get("depth", UDFProtocol.TILE_DEPTH_DEFAULT)
        if depth is UDFProtocol.TILE_DEPTH_DEFAULT:
            depth = 32
        if depth > approx_partition_shape[0]:
            depth = approx_partition_shape[0]
        return depth

    def _get_min_depth(self, udf: "UDFProtocol", approx_partition_shape: Shape) -> int:
        udf_method = udf.get_method()

        if udf_method == UDFMethod.PARTITION:
            return approx_partition_shape[0]
        elif udf_method == UDFMethod.TILE:
            return self._get_udf_depth_pref(udf, approx_partition_shape)
        return 1
