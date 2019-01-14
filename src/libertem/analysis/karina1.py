import functools
import operator

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

from .base import BaseAnalysis
from libertem.masks import radial_gradient, _make_circular_mask


class Karina1PhaseCorrelationAnalysis(BaseAnalysis):
    def get_parameters(self, parameters):
        params = {}
        params.update(parameters)
        return {
            'radius': parameters['radius'],
            'num_disks': parameters['num_disks'],
            'mask_type': parameters.get('mask_type', 'radial_gradient_1'),
            'padding': parameters.get('padding', 0),
            'enable_scaling': parameters.get('enable_scaling', True),
        }
        return params

    def __getstate__(self):
        # FIXME: ugly hack to check if pickle performance is bad:
        res = {}
        for k, v in self.__dict__.items():
            if k == "dataset":
                res[k] = None
            else:
                res[k] = v
        return res

    # NOTE: this is the entry point into the analysis
    def run_analysis(self, executor):
        # TODO: executor may be async, what do? it also may be sync... (how does dask do it?)
        # TODO: find a "lightweight" way to do init_result/compute/reduce_result?
        dataset = self.dataset
        sum_result_total = np.zeros(dataset.shape.sig, dtype="float32")
        for sum_result in executor.map_partitions(dataset=dataset, fn=self.pass_1):
            sum_result_total += sum_result

        peaks = self.get_peaks(
            framesize=tuple(dataset.shape.sig),
            sum_result=sum_result_total,
        )

        num_disks = self.parameters['num_disks']

        all_centers = np.zeros(tuple(dataset.raw_shape.nav) + (num_disks, 2), dtype="u2")
        all_peak_values = np.zeros(tuple(dataset.raw_shape.nav) + (num_disks,), dtype="float32")

        for centers, peak_values, partition in executor.map_partitions(
                dataset=dataset,
                fn=self.pass_2,
                peaks=peaks):
            slice_ = partition.slice.get(nav_only=True) + (...,)
            all_centers[slice_] = centers
            all_peak_values[slice_] = peak_values

        return sum_result_total, all_centers, all_peak_values, peaks

    def pass_1(self, partition):
        tiles = partition.get_tiles()  # full_frames=True)

        sum_result = np.zeros(partition.meta.shape.sig, dtype="float32")

        # TODO:
        # bf_result = np.zeros(partition.meta.shape.nav, dtype="float32")

        buf = np.zeros(partition.meta.shape.sig, dtype="float32")

        for tile in tiles:
            data = tile.flat_nav
            if data.dtype.kind in ('u', 'i'):
                data = data.astype("float32")
            for frame in data:
                sum_result += self.scale_frame(frame, out=buf)
                # TODO: calculate bright field image in parallel

        return sum_result

    def get_peaks(self, framesize, sum_result):
        """
        executed on master node, calculate crop rects from average image

        padding : float
            to prevent very close disks from interfering with another,
            we add only a small fraction of radius to area that will be cropped
        """
        parameters = self.parameters
        radius = parameters['radius']
        num_disks = parameters['num_disks']
        spec_mask = self.get_template(sig_shape=framesize, radius=radius)
        spec_sum = np.fft.rfft2(sum_result)
        corrspec = spec_mask * spec_sum
        corr = np.fft.fftshift(np.fft.irfft2(corrspec))
        peaks = peak_local_max(corr, num_peaks=num_disks)
        self._debug_template = spec_mask
        return peaks

    def pass_2(self, partition, peaks):
        parameters = self.parameters
        radius = parameters['radius']
        padding = parameters['padding']

        tiles = partition.get_tiles()  # full_frames=True)
        crop_size = self.get_crop_size(radius, padding)
        template = self.get_template(sig_shape=(2 * crop_size, 2 * crop_size), radius=radius)
        centers = np.zeros(tuple(partition.slice.shape.nav) + (len(peaks), 2),
                           dtype="u2")
        peak_values = np.zeros(tuple(partition.slice.shape.nav) + (len(peaks),),
                               dtype="float32")
        buf = np.zeros((2 * crop_size, 2 * crop_size), dtype="float32")
        for tile in tiles:
            data = tile.flat_nav
            tile_slice_in_partition = tile.tile_slice.shift(partition.slice)
            start_of_tile = np.ravel_multi_index(
                tile_slice_in_partition.origin[:-tile_slice_in_partition.shape.sig.dims],
                tuple(partition.shape.nav),
            )

            for frame_idx, frame in enumerate(data):
                for disk_idx, crop_part in enumerate(self.crop_disks_from_frame(
                        peaks=peaks,
                        frame=frame,
                        padding=padding,
                        radius=radius)):
                    if crop_part.size == 0:
                        continue
                        # raise Exception("crop part is zero-length for disk %d" % disk_idx)
                    # scaled = np.log(crop_part - np.min(crop_part) + 1, out=buf)
                    scaled = self.scale_frame(crop_part, out=buf)
                    center, peak_value = self.do_correlation(template, scaled)
                    crop_origin = peaks[disk_idx] - [crop_size, crop_size]
                    abs_center = tuple(center + crop_origin)

                    result_idx = np.unravel_index(start_of_tile + frame_idx,
                                                  partition.shape.nav) + (disk_idx,)
                    centers[result_idx] = abs_center
                    peak_values[result_idx] = peak_value
        return centers, peak_values, partition

    def get_crop_size(self, radius, padding):
        return int(radius + radius * padding)

    def crop_disks_from_frame(self, peaks, frame, padding, radius):
        crop_size = self.get_crop_size(radius, padding)
        for peak in peaks:
            yield frame[
                peak[0] - crop_size:peak[0] + crop_size,
                peak[1] - crop_size:peak[1] + crop_size,
            ]

    def scale_frame(self, frame, out):
        """
        scale/normalize a frame (or a part of a frame)
        """
        return np.log(frame - np.min(frame) + 1, out=out)

    def get_template(self, sig_shape, radius):
        def _make_circular_weighted_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
            mask = np.zeros([imageSizeY, imageSizeX])
            mask1 = np.zeros([imageSizeY, imageSizeX])
            for i in range(1, radius):
                xx, yy = circle_perimeter(centerX, centerY, i)
                mask[xx, yy] = (i)/radius
            ed = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius)
            mask[np.where((mask == 0) & (ed == 1))] = np.nan
            mask1 = pd.DataFrame(mask)
            mask = np.array(mask1.interpolate(method='linear'))
            return(mask)

        mask_type = self.parameters['mask_type']
        if mask_type == "radial_gradient_1":
            mask = _make_circular_weighted_mask(
                centerY=sig_shape[0] // 2,
                centerX=sig_shape[1] // 2,
                imageSizeY=sig_shape[0],
                imageSizeX=sig_shape[1],
                radius=radius,
            )
        elif mask_type == "radial_gradient_2":
            mask = radial_gradient(
                centerY=sig_shape[0] // 2,
                centerX=sig_shape[1] // 2,
                imageSizeY=sig_shape[0],
                imageSizeX=sig_shape[1],
                radius=radius,
            )
        else:
            raise ValueError("unknown mask type: %s" % mask_type)

        spec_mask = np.fft.rfft2(mask)
        return spec_mask

    def do_correlation(self, template, crop_part):
        spec_part = np.fft.rfft2(crop_part)
        corrspec = template * spec_part
        corr = np.fft.fftshift(np.fft.irfft2(corrspec))
        center = np.unravel_index(np.argmax(corr), corr.shape)
        return center, corr[center]
