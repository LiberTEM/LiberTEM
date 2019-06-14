import numpy as np


from utils import MemoryDataSet, _mk_random


def test_partition3d_correct_slices():
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 16, 16),
                            num_partitions=2, sig_dims=2)
    mask = np.zeros(data.shape[:2], dtype=bool)
    mask[0, 0] = True
    mask[15, 0] = True

    partitions = dataset.get_partitions()
    p1 = next(partitions)
    p2 = next(partitions)

    assert len(list(p1.get_tiles(roi=mask))) == 1
    assert len(list(p2.get_tiles(roi=mask))) == 1

    t1 = next(p1.get_tiles(roi=mask))
    t2 = next(p2.get_tiles(roi=mask))

    print("t1", t1.tile_slice)
    print("t2", t2.tile_slice)

    assert t1.tile_slice.origin[0] == 0
    assert t2.tile_slice.origin[0] == 1
