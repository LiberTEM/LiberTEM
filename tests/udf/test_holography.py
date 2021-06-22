import pytest
import numpy as np

from libertem.udf.holography import HoloReconstructUDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.generate import hologram_frame
from libertem.utils.devices import detect
from libertem.common.backend import set_use_cpu, set_use_cuda


@pytest.mark.parametrize(
    # CuPy support deactivated due to https://github.com/LiberTEM/LiberTEM/issues/815
    # 'backend', ['numpy', 'cupy']
    'backend', ['numpy']
)
def test_holo_reconstruction(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    # Prepare image parameters and mesh
    nx, ny = (5, 7)
    sx, sy = (64, 64)
    slice_crop = (slice(None),
                  slice(None),
                  slice(sx // 4, sx // 4 * 3),
                  slice(sy // 4, sy // 4 * 3))

    lnx = np.arange(nx)
    lny = np.arange(ny)
    lsx = np.arange(sx)
    lsy = np.arange(sy)

    mnx, mny, msx, msy = np.meshgrid(lnx, lny, lsx, lsy)

    # Prepare phase image
    phase_ref = np.pi * msx * (mnx.max() - mnx) * mny / sx**2 \
        + np.pi * msy * mnx * (mny.max() - mny) / sy**2

    # Generate holograms
    holo = np.zeros_like(phase_ref)
    ref = np.zeros_like(phase_ref)

    for i in range(nx):
        for j in range(ny):
            holo[j, i, :, :] = hologram_frame(np.ones((sx, sy)), phase_ref[j, i, :, :])
            ref[j, i, :, :] = hologram_frame(np.ones((sx, sy)), np.zeros((sx, sy)))

    # Prepare LT datasets and do reconstruction
    dataset_holo = MemoryDataSet(data=holo, tileshape=(ny, sx, sy),
                                 num_partitions=2, sig_dims=2)

    dataset_ref = MemoryDataSet(data=ref, tileshape=(ny, sx, sy),
                                num_partitions=1, sig_dims=2)

    sb_position = [11, 6]
    sb_size = 6.26498204

    holo_job = HoloReconstructUDF(out_shape=(sx, sy),
                                  sb_position=sb_position,
                                  sb_size=sb_size)
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        w_holo = lt_ctx.run_udf(dataset=dataset_holo, udf=holo_job)['wave'].data
        w_ref = lt_ctx.run_udf(dataset=dataset_ref, udf=holo_job)['wave'].data
    finally:
        set_use_cpu(0)

    w = w_holo / w_ref

    phase = np.angle(w)

    assert np.allclose(phase_ref[slice_crop], phase[slice_crop], rtol=0.12)


def test_sb_pos_invalid(lt_ctx):
    with pytest.raises(ValueError) as e:
        HoloReconstructUDF(
            out_shape=(128, 128),
            sb_position=(0, 1, 2),
            sb_size=(0, 1, 2)
        )

    assert e.match(r"invalid sb_position \(0, 1, 2\), must be tuple of length 2")
