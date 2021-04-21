# Functions freq_array, aperture_function, are adopted from Hyperspy
# and are subject of following copyright:
#
#  Copyright 2007-2016 The HyperSpy developers
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2019 The LiberTEM developers
#
#  LiberTEM is distributed under the terms of the GNU General
# Public License as published by the Free Software Foundation,
# version 3 of the License.
# see: https://github.com/LiberTEM/LiberTEM

import numpy as np

from libertem.udf import UDF


def freq_array(shape, sampling=(1., 1.)):
    """
    Makes up a frequency array.

    Parameters
    ----------
    shape : (int, int)
        The shape of the array.
    sampling: (float, float), optional, (Default: (1., 1.))
        The sampling rates of the array.
    Returns
    -------
        Array of the frequencies.
    """
    f_freq_1d_y = np.fft.fftfreq(shape[0], sampling[0])
    f_freq_1d_x = np.fft.fftfreq(shape[1], sampling[1])
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y)
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq


def aperture_function(r, apradius, rsmooth):
    """
    A smooth aperture function that decays from apradius-rsmooth to apradius+rsmooth.
    Parameters
    ----------
    r : ndarray
        Array of input data (e.g. frequencies)
    apradius : float
        Radius (center) of the smooth aperture. Decay starts at apradius - rsmooth.
    rsmooth : float
        Smoothness in halfwidth. rsmooth = 1 will cause a decay from 1 to 0 over 2 pixel.
    Returns
    -------
        2d array containing aperture
    """

    return 0.5 * (1. - np.tanh((np.absolute(r) - apradius) / (0.5 * rsmooth)))


class HoloReconstructUDF(UDF):
    """
    Reconstruct off-axis electron holograms using a Fourier-based method.

    Running :meth:`~libertem.api.Context.run_udf` on an instance of this class
    will reconstruct a complex electron wave. Use the :code:`wave` key to access
    the raw data in the result.

    See :ref:`holography app` for detailed application example

    .. versionadded:: 0.3.0

    Parameters
    ----------

    out_shape : (int, int)
        Shape of the returned complex wave image. Note that the result should fit into the
        main memory.
        See :ref:`holography app` for more details

    sb_position : tuple, or vector
        Coordinates of sideband position with respect to non-shifted FFT of a hologram

    sb_size : float
        Radius of side band filter in pixels

    sb_smoothness : float, optional (Default: 0.05)
        Fraction of `sb_size` over which the edge of the filter aperture to be smoothed

    precision : bool, optional, (Default: True)
        Defines precision of the reconstruction, True for complex128 for the resulting
        complex wave, otherwise results will be complex64

    Examples
    --------
    >>> shape = tuple(dataset.shape.sig)
    >>> sb_position = [2, 3]
    >>> sb_size = 4.4
    >>> holo_udf = HoloReconstructUDF(out_shape=shape,
    ...                               sb_position=sb_position,
    ...                               sb_size=sb_size)
    >>> wave = ctx.run_udf(dataset=dataset, udf=holo_udf)['wave'].data
    """
    def __init__(self,
                 out_shape,
                 sb_position,
                 sb_size,
                 sb_smoothness=.05,
                 precision=True):
        if len(sb_position) != 2:
            raise ValueError("invalid sb_position %r, must be tuple of length 2" % (sb_position,))
        super().__init__(out_shape=out_shape,
                         sb_position=sb_position,
                         sb_size=sb_size,
                         sb_smoothness=sb_smoothness,
                         precision=precision)

    def get_result_buffers(self):
        """
        Initializes :class:`~libertem.common.buffers.BufferWrapper` objects for reconstructed
        wave function

        Returns
        -------
        A dictionary that maps 'wave' to the corresponding
        :class:`~libertem.common.buffers.BufferWrapper` objects
        """
        extra_shape = self.params.out_shape
        if not self.params.precision:
            dtype = np.complex64
        else:
            dtype = np.complex128
        return {
            "wave": self.buffer(kind="nav", dtype=dtype, extra_shape=extra_shape)
        }

    def get_task_data(self):
        """
        Updates `task_data`

        Returns
        -------
        kwargs : dict
        A dictionary with the following keys:
            kwargs['aperture'] : array-like
            Side band filter aperture (mask)
            kwargs['slice'] : slice
            Slice for slicing FFT of the hologram
        """

        out_shape = self.params.out_shape
        sy, sx = self.meta.partition_shape.sig
        oy, ox = out_shape
        f_sampling = (1. / oy, 1. / ox)
        sb_size = self.params.sb_size * np.mean(f_sampling)
        sb_smoothness = sb_size * self.params.sb_smoothness * np.mean(f_sampling)

        f_freq = freq_array(out_shape)
        aperture = aperture_function(f_freq, sb_size, sb_smoothness)

        y_min = int(sy / 2 - oy / 2)
        y_max = int(sy / 2 + oy / 2)
        x_min = int(sx / 2 - ox / 2)
        x_max = int(sx / 2 + oy / 2)
        slice_fft = (slice(y_min, y_max), slice(x_min, x_max))

        kwargs = {
            'aperture': self.xp.array(aperture),
            'slice': slice_fft
        }
        return kwargs

    def process_frame(self, frame):
        """
        Reconstructs holograms outputting results into 'wave'

        Parameters
        ----------
        frame
           single frame (hologram) of the data
        """
        if not self.params.precision:
            frame = frame.astype(np.float32)
        # size_x, size_y = self.params.out_shape
        frame_size = self.meta.partition_shape.sig
        sb_pos = self.params.sb_position
        aperture = self.task_data.aperture
        slice_fft = self.task_data.slice

        fft_frame = self.xp.fft.fft2(frame) / np.prod(frame_size)
        fft_frame = self.xp.roll(fft_frame, sb_pos, axis=(0, 1))

        fft_frame = self.xp.fft.fftshift(self.xp.fft.fftshift(fft_frame)[slice_fft])

        fft_frame = fft_frame * aperture

        wav = self.xp.fft.ifft2(fft_frame) * np.prod(frame_size)
        # FIXME check if result buffer with where='device' and export is faster
        # than exporting frame by frame, as implemented now.
        if self.meta.device_class == 'cuda':
            # That means xp is cupy
            wav = self.xp.asnumpy(wav)
        self.results.wave[:] = wav

    def get_backends(self):
        # CuPy support deactivated due to https://github.com/LiberTEM/LiberTEM/issues/815
        return ('numpy',)
        # return ('numpy', 'cupy')
