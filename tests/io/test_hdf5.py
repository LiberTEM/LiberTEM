import pickle
import cloudpickle
import numpy as np
from libertem.io.dataset.hdf5 import H5DataSet

from utils import _naive_mask_apply, _mk_random


def test_hdf5_apply_masks_1(lt_ctx, hdf5_ds_1):
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)
    analysis = lt_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )


def test_pickle_ds(lt_ctx, hdf5_ds_1):
    pickled = pickle.dumps(hdf5_ds_1)
    loaded = pickle.loads(pickled)

    assert loaded._dtype is not None
    assert loaded._raw_shape is not None

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 1 * 1024


def test_cloudpickle(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16), target_size=512*1024*1024
    )

    pickled = cloudpickle.dumps(ds)
    loaded = cloudpickle.loads(pickled)

    assert loaded._dtype is None
    assert loaded._raw_shape is None
    repr(loaded)

    ds.initialize()

    pickled = cloudpickle.dumps(ds)
    loaded = cloudpickle.loads(pickled)

    assert loaded._dtype is not None
    assert loaded._raw_shape is not None
    loaded.shape
    loaded.dtype
    repr(loaded)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 1 * 1024
