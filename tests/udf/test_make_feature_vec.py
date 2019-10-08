import numpy as np

import libertem.udf.feature_vector_maker as feature

from libertem.io.dataset.memory import MemoryDataSet


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
    result, coordinates = feature.make_feature_vec(ctx=lt_ctx, dataset=dataset,
    delta=0, n_peaks=5, min_dist=0)
    # check if values of feature vectors are zeros for amorphous frames
    assert np.allclose(result['feature_vec'].data[6:9], np.zeros([3, 4]))
    # check if all values of feature vectors are NOT zeros for strong crystalline frames
    # (at least one peak is recognized)
    assert (result['feature_vec'].data[0:3] > np.zeros([3, 4])).any()
    # check if all values of feature vector are NOT zeros for weak crystalline frames
    # (at least one peak is recognized)
    assert (result['feature_vec'].data[3:6] > np.zeros([3, 4])).any()
    # check of feature vectors are NOT equal for strong crystalline frames
    #  than for weak crystalline frames
    # (because of non-zero order diffraction peaks are in different positions)
    assert (result['feature_vec'].data[3:6] != result['feature_vec'].data[0:3]).all()
