import numpy as np

import libertem.udf.crystallinity as crystal
from libertem.analysis.apply_fft_mask import ApplyFFTMask

from libertem.io.dataset.memory import MemoryDataSet


def test_smoke(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    result = crystal.run_analysis_crystall(
        ctx=lt_ctx, dataset=dataset, rad_in=0, rad_out=1,
        real_center=(1, 1), real_rad=1)
    assert np.allclose(result['intensity'].data, np.zeros(data.shape[0]))


def test_simple_example(lt_ctx):
    # creating a dataset where 0:3 frames are strong crystalline, 3:6 frames are weak crystalline,
    #  6:9 frames are amourphous
    data = np.zeros([3*3, 5, 5]).astype(np.float32)
    # adding high intensity zero order peak for all frames
    data[:, 2, 2] = 7
    # adding strong non-zero order diffraction peaks for 0:3 frames
    data[0:3, 0, 0] = 2
    data[0:3, 4, 4] = 2
    # adding weak non-zero order diffraction peaks for 0:3 frames
    data[3:6, 2, 0] = 1
    data[3:6, 2, 4] = 1
    dataset = MemoryDataSet(data=data, tileshape=(1, 5, 5),
                            num_partitions=3, sig_dims=2)
    result = crystal.run_analysis_crystall(ctx=lt_ctx, dataset=dataset, rad_in=0, rad_out=3,
                                           real_center=(2, 2), real_rad=0)
    # check if values of integration in Fourier space after deleting of zero order diffraction peaks
    #  are zeros for amorphous frames
    assert np.allclose(result['intensity'].data[6:9], np.zeros([3]))
    # check if values of integration in Fourier space after deleting of zero order diffraction peaks
    #  are NOT zeros for strong crystalline frames
    assert (result['intensity'].data[0:3] > np.zeros([3])).all()
    # check if values of integration in Fourier space after deleting of zero order diffraction peaks
    #  are NOT zeros for weak crystalline frames
    assert (result['intensity'].data[3:6] > np.zeros([3])).all()
    # check if values of integration in Fourier space after deleting of zero order diffraction peaks
    #  are higher for strong crystalline frames than for weak crystalline frames
    assert (result['intensity'].data[0:3] > result['intensity'].data[3:6]).all()


def test_fft_mask_analysis(lt_ctx):
    # same setup as above
    data = np.zeros([3*3, 5, 5]).astype(np.float32)
    data[:, 2, 2] = 7
    data[0:3, 0, 0] = 2
    data[0:3, 4, 4] = 2
    data[3:6, 2, 0] = 1
    data[3:6, 2, 4] = 1
    dataset = MemoryDataSet(data=data, tileshape=(1, 5, 5),
                            num_partitions=3, sig_dims=2)
    analysis = ApplyFFTMask(dataset=dataset, parameters={
        'rad_in': 0,
        'rad_out': 3,
        'real_centerx': 2,
        'real_centery': 2,
        'real_rad': 0,
    })
    result = lt_ctx.run(analysis)

    assert np.allclose(result['intensity'].raw_data[6:9], np.zeros([3]))
    assert (result['intensity'].raw_data[0:3] > np.zeros([3])).all()
    assert (result['intensity'].raw_data[3:6] > np.zeros([3])).all()
    assert (result['intensity'].raw_data[0:3] > result['intensity'].raw_data[3:6]).all()
