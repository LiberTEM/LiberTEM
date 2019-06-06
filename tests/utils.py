import numpy as np

from libertem.io.dataset.base import (
    FileSet3D, Partition3D, DataSet, DataSetMeta
)
from libertem.common import Shape
from libertem.masks import to_dense


class MemoryReader(object):
    def __init__(self, data):
        self.data = data


class MemoryFile3D(object):
    def __init__(self, data, check_cast=True):
        self.num_frames = data.shape[0]
        self.start_idx = 0
        self.end_idx = self.num_frames
        self._data = data
        self._check_cast = check_cast

    def open(self):
        pass

    def close(self):
        pass

    def readinto(self, start, stop, out, crop_to=None):
        slice_ = (...,)
        if crop_to is not None:
            slice_ = crop_to.get(sig_only=True)
        if self._check_cast:
            assert np.can_cast(self._data.dtype, out.dtype, casting='safe'),\
                "cannot cast safely between %s and %s" % (self._data.dtype, out.dtype)
        out[:] = self._data[(slice(start, stop),) + slice_]


class MemoryDataSet(DataSet):
    def __init__(self, data, tileshape, num_partitions, sig_dims=2, check_cast=True):
        assert len(tileshape) == sig_dims + 1
        self.data = data
        self.tileshape = Shape(tileshape, sig_dims=sig_dims)
        self.num_partitions = num_partitions
        self.sig_dims = sig_dims
        self._meta = DataSetMeta(
            shape=self.shape,
            raw_dtype=self.data.dtype,
        )
        self._check_cast = check_cast

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return Shape(self.data.shape, sig_dims=self.sig_dims)

    def check_valid(self):
        return True

    def get_reader(self):
        return MemoryReader(data=self.data)

    def get_partitions(self):
        fileset = FileSet3D([
            MemoryFile3D(self.data.reshape(self.shape.flatten_nav()),
                         check_cast=self._check_cast)
        ])

        stackheight = int(np.product(self.tileshape[:-self.sig_dims]))
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self.num_partitions):
            print("creating partition", part_slice, start, stop, stackheight)
            yield Partition3D(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset.get_for_range(start, stop),
                start_frame=start,
                num_frames=stop - start,
                stackheight=stackheight,
            )


def _naive_mask_apply(masks, data):
    """
    masks: list of masks
    data: 4d array of input data

    returns array of shape (num_masks, scan_y, scan_x)
    """
    assert len(data.shape) == 4
    for mask in masks:
        assert mask.shape == data.shape[2:], "mask doesn't fit frame size"

    dtype = np.result_type(*masks, data)
    res = np.zeros((len(masks),) + tuple(data.shape[:2]), dtype=dtype)
    for n in range(len(masks)):
        mask = to_dense(masks[n])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = data[i, j].ravel().dot(mask.ravel())
                res[n, i, j] = item
    return res


# This function introduces asymmetries so that errors won't average out so
# easily with large data sets
def _mk_random(size, dtype='float32'):
    dtype = np.dtype(dtype)
    if dtype.kind == 'c':
        choice = [0, 1, -1, 0+1j, 0-1j]
    else:
        choice = [0, 1]
    data = np.random.choice(choice, size=size).astype(dtype)
    coords2 = tuple((np.random.choice(range(c)) for c in size))
    coords10 = tuple((np.random.choice(range(c)) for c in size))
    data[coords2] = np.random.choice(choice) * sum(size)
    data[coords10] = np.random.choice(choice) * 10 * sum(size)
    return data


def assert_msg(msg, msg_type, status='ok'):
    print(msg, msg_type, status)
    assert msg['status'] == status
    assert msg['messageType'] == msg_type,\
        "expected: {}, is: {}".format(msg_type, msg['messageType'])
