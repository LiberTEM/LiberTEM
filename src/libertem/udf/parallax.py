import numpy as np

import numba
from libertem.udf import UDF


def electron_wavelength_np(energy: float):
    m = 9.109383e-31
    e = 1.602177e-19
    c = 299792458
    h = 6.62607e-34

    lam = h / np.sqrt(2 * m * e * energy) / np.sqrt(1 + e * energy / 2 / m / c**2)
    return lam * 1e10


def spatial_frequencies_np(gpts, sampling, rotation_angle=None):
    """ """
    ny, nx = gpts
    sy, sx = sampling

    kx = np.fft.fftfreq(ny, sy)
    ky = np.fft.fftfreq(nx, sx)
    kxa, kya = np.meshgrid(kx, ky, indexing="ij")

    if rotation_angle is not None:
        c = np.cos(rotation_angle)
        s = np.sin(rotation_angle)
        kx_rot = c * kxa - s * kya
        ky_rot = s * kxa + c * kya
        kxa, kya = kx_rot, ky_rot

    return kxa, kya


def polar_coordinates_np(kx, ky):
    """ """
    k = np.sqrt(kx**2 + ky**2)
    phi = np.arctan2(ky, kx)
    return k, phi


def aberration_surface_np(alpha, phi, wavelength, aberration_coefs):
    """ """
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)

    prefactor = np.pi / wavelength

    chi = prefactor * alpha**2 * (C10 + C12 * np.cos(2.0 * (phi - phi12)))

    return chi


def aberration_cartesian_gradients_np(alpha, phi, aberration_coefs):
    """ """
    C10 = aberration_coefs.get("C10", 0.0)
    C12 = aberration_coefs.get("C12", 0.0)
    phi12 = aberration_coefs.get("phi12", 0.0)

    cos2 = np.cos(2.0 * (phi - phi12))
    sin2 = np.sin(2.0 * (phi - phi12))

    # dχ/dα and dχ/dφ
    scale = 2 * np.pi
    dchi_dk = scale * alpha * (C10 + C12 * cos2)
    dchi_dphi = -scale * alpha * (C12 * sin2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    dchi_dx = cos_phi * dchi_dk - sin_phi * dchi_dphi
    dchi_dy = sin_phi * dchi_dk + cos_phi * dchi_dphi

    return dchi_dx, dchi_dy


def suppress_nyquist_realspace(array):
    """
    Zero Nyquist frequencies of a real-space array.
    """
    F = np.fft.fft2(array)
    Nx, Ny = F.shape
    F[Nx // 2, :] = 0.0
    F[:, Ny // 2] = 0.0
    return np.fft.ifft2(F).real


@numba.njit(fastmath=True, nogil=True, cache=False)
def parallax_accumulate_cpu(
    frames,  # (T, sy, sx) float32/64
    bf_flat_inds,  # (M,) int32
    shifts,  # (M,2) int32
    coords,  # (T,2) int64
    out,  # (Ny, Nx) float64
):
    """
    Real-valued parallax accumulation with per-frame mean subtraction.
    """
    T = frames.shape[0]
    M = shifts.shape[0]
    Ny, Nx = out.shape
    # sy = frames.shape[1]
    sx = frames.shape[2]

    for t in range(T):
        frame = frames[t]

        mean = 0.0
        for m in range(M):
            flat_idx = bf_flat_inds[m]
            iy = flat_idx // sx
            ix = flat_idx % sx
            mean += frame[iy, ix]
        mean /= M

        yt, xt = coords[t]

        mean
        for m in range(M):
            flat_idx = bf_flat_inds[m]
            iy = flat_idx // sx
            ix = flat_idx % sx

            val = frame[iy, ix] - mean

            sy_m, sx_m = shifts[m]
            oy = (yt + sy_m) % Ny
            ox = (xt + sx_m) % Nx

            out[oy, ox] += val


class ParallaxUDF(UDF):
    """
    LiberTEM User-Defined Function for streaming parallax reconstruction,
    with internal NumPy preprocessing equivalent to StreamingParallax.
    """

    def __init__(
        self, bf_flat_inds, shifts, upsampling_factor, suppress_Nyquist_noise, **kwargs
    ):
        super().__init__(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            upsampling_factor=upsampling_factor,
            suppress_Nyquist_noise=suppress_Nyquist_noise,
            **kwargs,
        )

    @classmethod
    def from_parameters(
        cls,
        gpts: tuple[int, int],
        scan_gpts: tuple[int, int],
        scan_sampling: tuple[float, float],
        energy: float,
        semiangle_cutoff: float,
        reciprocal_sampling: tuple[float, float] | None = None,
        angular_sampling: tuple[float, float] | None = None,
        aberration_coefs: dict | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
        suppress_Nyquist_noise: bool = True,
        **kwargs,
    ):
        """ """

        wavelength = electron_wavelength_np(energy)
        if angular_sampling is not None:
            if reciprocal_sampling is not None:
                raise ValueError(
                    "Only one of reciprocal_sampling / angular_sampling can be specified"
                )
            reciprocal_sampling = tuple(a / wavelength / 1e3 for a in angular_sampling)

        aberration_coefs = aberration_coefs or {}
        ny, nx = gpts

        # ---- Parallax shifts ----
        sampling = (
            1.0 / reciprocal_sampling[0] / ny,  # ty:ignore[not-subscriptable]
            1.0 / reciprocal_sampling[1] / nx,  # ty:ignore[not-subscriptable]
        )

        kxa, kya = spatial_frequencies_np(
            gpts,
            sampling,
            rotation_angle=rotation_angle,
        )
        k, phi = polar_coordinates_np(kxa, kya)

        # ---- BF indices ----
        bf_mask = k * wavelength * 1e3 <= semiangle_cutoff
        inds_i, inds_j = np.where(bf_mask)

        inds_i_fft = (inds_i - ny // 2) % ny
        inds_j_fft = (inds_j - nx // 2) % nx
        bf_flat_inds = (inds_i_fft * nx + inds_j_fft).astype(np.int32)

        dx, dy = aberration_cartesian_gradients_np(
            k * wavelength,
            phi,
            aberration_coefs,
        )

        grad_k = np.stack(
            (dx[inds_i, inds_j], dy[inds_i, inds_j]),
            axis=-1,
        )

        upsampled_sampling = (
            scan_sampling[0] / upsampling_factor,
            scan_sampling[1] / upsampling_factor,
        )

        shifts = np.round(grad_k / (2 * np.pi) / upsampled_sampling).astype(np.int32)

        return cls(
            bf_flat_inds=bf_flat_inds,
            shifts=shifts,
            upsampling_factor=upsampling_factor,
            suppress_Nyquist_noise=suppress_Nyquist_noise,
            **kwargs,
        )

    def get_result_buffers(self):
        return {
            "reconstruction": self.buffer(
                kind="single",
                dtype=np.float64,
                extra_shape=self.upsampled_scan_gpts,
            )
        }

    @property
    def gpts(self):
        return self.meta.dataset_shape.sig

    @property
    def scan_gpts(self):
        return self.meta.dataset_shape.nav

    @property
    def upsampled_scan_gpts(self) -> tuple[int, int]:
        upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]
        return tuple(gpt * upsampling_factor for gpt in self.scan_gpts)

    def process_tile(self, tile):
        frames = tile.data
        coords = self.meta.coordinates

        upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]
        coords *= upsampling_factor

        parallax_accumulate_cpu(
            frames,
            self.params.bf_flat_inds,
            self.params.shifts,
            coords,
            self.results.reconstruction,
        )

    def merge(self, dest, src):
        reconstruction = src.reconstruction
        upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]

        if self.params.suppress_Nyquist_noise and upsampling_factor > 1:
            reconstruction = suppress_nyquist_realspace(reconstruction)

        dest.reconstruction[:] += reconstruction

    def postprocess(self):
        pass


def prepare_grouped_phase_flipping_kernel_np(H, s_m_up, upsampled_gpts):
    """
    Prepare the phase-flip kernel offsets and weights in NumPy.

    Parameters
    ----------
    H : np.ndarray, shape (h, w)
        Base kernel.
    s_m_up : np.ndarray, shape (M, 2)
        Up-sampled shifts for M BF pixels, [y, x].
    upsampled_gpts : tuple[int, int]
        (Ny, Nx) real-space grid size

    Returns
    -------
    unique_offsets : np.ndarray[int64], shape (U,)
        Flattened offsets for scatter-add.
    K : np.ndarray[float64], shape (U, M)
        Phase-flip weights for each unique offset and BF pixel.
    """
    Ny, Nx = upsampled_gpts
    h, w = H.shape
    M = s_m_up.shape[0]
    L0 = h * w

    # kernel grid
    dy = np.arange(h)
    dx = np.arange(w)
    dy_grid = np.repeat(dy, w)
    dx_grid = np.tile(dx, h)

    # repeat for M BF pixels
    dy_rep = np.tile(dy_grid, M)
    dx_rep = np.tile(dx_grid, M)

    # shifts repeated
    s_my = np.repeat(s_m_up[:, 0], L0)
    s_mx = np.repeat(s_m_up[:, 1], L0)

    # compute flattened offsets
    offsets = (dy_rep + s_my) * Nx + (dx_rep + s_mx)

    # find unique offsets and inverse indices
    unique_offsets, inv = np.unique(offsets, return_inverse=True)
    U = unique_offsets.size

    # build grouped kernel
    H_flat = H.ravel()
    H_all = np.tile(H_flat, M)
    m_idx = np.repeat(np.arange(M), L0)

    K = np.zeros((U, M), dtype=H.dtype)
    np.add.at(K, (inv, m_idx), H_all)  # accumulate values

    return unique_offsets.astype(np.int64), K


@numba.njit(fastmath=True, nogil=True, cache=False)
def phase_flip_accumulate_cpu(frames, bf_rows, bf_cols, coords, unique_offsets, K, out):
    """
    Scatter-add phase-flip contributions into out (real-space accumulator).

    Parameters
    ----------
    frames : (T, sy, sx) float32/64
        Input frames (BFS pixels)
    bf_rows, bf_cols : (M,) int32
        Row/col indices of BF pixels
    coords : (T, 2) int64
        Real-space navigation coordinates
    unique_offsets : (U,) int64
        Flattened offsets for the phase-flip kernel
    K : (U, M) float64
        Phase-flip weights
    out : (Ny, Nx) float64
        Real-space accumulator
    """
    T = frames.shape[0]
    M = len(bf_rows)
    U = len(unique_offsets)
    Ny, Nx = out.shape

    for t in range(T):
        # Extract BF pixels and subtract per-frame mean
        I_bf = np.empty(M, dtype=np.float64)
        s = 0.0
        for m in range(M):
            val = frames[t, bf_rows[m], bf_cols[m]]
            s += val
            I_bf[m] = val
        mean = s / M
        for m in range(M):
            I_bf[m] -= mean

        # Compute contributions
        vals = np.empty(U, dtype=np.float64)
        for u in range(U):
            acc = 0.0
            for m in range(M):
                acc += K[u, m] * I_bf[m]
            vals[u] = acc

        # Scatter-add to accumulator
        yt, xt = coords[t]
        r_off = yt * Nx + xt
        for u in range(U):
            idx = (r_off + unique_offsets[u]) % (Ny * Nx)
            out.flat[idx] += vals[u]


class ParallaxPhaseFlipUDF(UDF):
    """
    LiberTEM UDF for streaming parallax with real-valued phase-flip operator.

    Parameters
    ----------
    bf_flat_inds : np.ndarray[int32]
        Flat indices of BF pixels in FFT-native layout.
    unique_offsets : np.ndarray[int64]
        Flattened offsets corresponding to the phase-flip kernel.
    K : np.ndarray[float64]
        Phase-flip weights (shape U x M).
    """

    def __init__(self, bf_flat_inds, unique_offsets, K, upsampling_factor, **kwargs):
        super().__init__(
            bf_flat_inds=bf_flat_inds,
            unique_offsets=unique_offsets,
            K=K,
            upsampling_factor=upsampling_factor,
            **kwargs,
        )

    @classmethod
    def from_parameters(
        cls,
        gpts: tuple[int, int],
        scan_gpts: tuple[int, int],
        scan_sampling: tuple[float, float],
        energy: float,
        semiangle_cutoff: float,
        reciprocal_sampling: tuple[float, float] | None = None,
        angular_sampling: tuple[float, float] | None = None,
        aberration_coefs: dict | None = None,
        rotation_angle: float | None = None,
        upsampling_factor: int = 1,
        **kwargs,
    ):
        """ """
        wavelength = electron_wavelength_np(energy)
        if angular_sampling is not None:
            if reciprocal_sampling is not None:
                raise ValueError(
                    "Only one of reciprocal_sampling / angular_sampling can be specified"
                )
            reciprocal_sampling = tuple(a / wavelength / 1e3 for a in angular_sampling)

        aberration_coefs = aberration_coefs or {}
        ny, nx = gpts

        # ---- Parallax shifts ----
        sampling = (
            1.0 / reciprocal_sampling[0] / ny,  # ty:ignore[not-subscriptable]
            1.0 / reciprocal_sampling[1] / nx,  # ty:ignore[not-subscriptable]
        )

        kxa, kya = spatial_frequencies_np(
            gpts,
            sampling,
            rotation_angle=rotation_angle,
        )
        k, phi = polar_coordinates_np(kxa, kya)

        # ---- BF indices ----
        bf_mask = k * wavelength * 1e3 <= semiangle_cutoff
        inds_i, inds_j = np.where(bf_mask)

        inds_i_fft = (inds_i - ny // 2) % ny
        inds_j_fft = (inds_j - nx // 2) % nx
        bf_flat_inds = (inds_i_fft * nx + inds_j_fft).astype(np.int32)

        dx, dy = aberration_cartesian_gradients_np(
            k * wavelength,
            phi,
            aberration_coefs,
        )

        grad_k = np.stack(
            (dx[inds_i, inds_j], dy[inds_i, inds_j]),
            axis=-1,
        )

        upsampled_gpts = (
            scan_gpts[0] * upsampling_factor,
            scan_gpts[1] * upsampling_factor,
        )
        upsampled_sampling = (
            scan_sampling[0] / upsampling_factor,
            scan_sampling[1] / upsampling_factor,
        )

        shifts = np.round(grad_k / (2 * np.pi) / upsampled_sampling).astype(np.int32)

        qxa, qya = spatial_frequencies_np(
            upsampled_gpts,
            upsampled_sampling,
        )
        q, theta = polar_coordinates_np(qxa, qya)
        chi_q = aberration_surface_np(
            q * wavelength,
            theta,
            wavelength,
            aberration_coefs=aberration_coefs,
        )
        sign_sin_chi_q = np.sign(np.sin(chi_q))

        Nx, Ny = sign_sin_chi_q.shape
        sign_sin_chi_q[Nx // 2, :] = 0.0
        sign_sin_chi_q[:, Ny // 2] = 0.0
        H = np.fft.ifft2(sign_sin_chi_q).real

        unique_offsets, K = prepare_grouped_phase_flipping_kernel_np(
            H, shifts, upsampled_gpts
        )

        return cls(
            bf_flat_inds=bf_flat_inds,
            unique_offsets=unique_offsets,
            K=K,
            upsampling_factor=upsampling_factor,
            **kwargs,
        )

    def get_result_buffers(self):
        return {
            "reconstruction": self.buffer(
                kind="single",
                dtype=np.float64,
                extra_shape=self.upsampled_scan_gpts,
            )
        }

    @property
    def gpts(self):
        return self.meta.dataset_shape.sig

    @property
    def scan_gpts(self):
        return self.meta.dataset_shape.nav

    @property
    def upsampled_scan_gpts(self) -> tuple[int, int]:
        upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]
        return tuple(gpt * upsampling_factor for gpt in self.scan_gpts)

    def process_tile(self, tile):
        frames = tile.data  # shape (T, sy, sx)
        coords = self.meta.coordinates  # shape (T, 2)

        # Note we don't need to scale coords by upsampling-factor here, would be double-counting
        # upsampling_factor: int = self.params.upsampling_factor  # ty:ignore[invalid-assignment]
        # coords *= upsampling_factor

        bf_flat_inds = np.asarray(self.params.bf_flat_inds)
        bf_rows = bf_flat_inds // self.gpts[0]
        bf_cols = bf_flat_inds % self.gpts[0]

        phase_flip_accumulate_cpu(
            frames,
            bf_rows,
            bf_cols,
            coords,
            self.params.unique_offsets,
            self.params.K,
            self.results.reconstruction,
        )

    def merge(self, dest, src):
        dest.reconstruction[:] += src.reconstruction

    def postprocess(self):
        # No extra post-processing needed
        pass
