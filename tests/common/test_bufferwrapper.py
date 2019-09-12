import numpy as np

from libertem.io.dataset.memory import MemoryDataSet
from libertem.common.buffers import AuxBufferWrapper

from utils import _mk_random


def test_new_for_partition():
    auxdata = _mk_random(size=(16, 16), dtype="float32")
    buf = AuxBufferWrapper(kind="nav", dtype="float32")
    buf.set_buffer(auxdata)

    dataset = MemoryDataSet(data=_mk_random(size=(16, 16, 16, 16), dtype="float32"),
                            tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    assert auxdata.shape == tuple(dataset.shape.nav)

    roi = _mk_random(size=dataset.shape.nav, dtype="bool")

    for idx, partition in enumerate(dataset.get_partitions()):
        print("partition number", idx)
        new_buf = buf.new_for_partition(partition, roi=roi)
        ps = partition.slice.get(nav_only=True)
        roi_part = roi.reshape(-1)[ps]

        assert np.product(new_buf._data.shape) == roi_part.sum()

        # old buffer stays the same:
        assert np.allclose(buf._data, auxdata.reshape(-1))
        assert buf._data_coords_global
        assert not new_buf._data_coords_global

        # new buffer is sliced to partition and has ROI applied:
        assert new_buf._data.shape[0] <= buf._data.shape[0]
        assert new_buf._data.shape[0] <= partition.shape[0]

        # let's try and manually apply the ROI to `auxdata`:
        assert np.allclose(
            new_buf._data,
            auxdata.reshape(-1)[ps][roi_part]
        )
