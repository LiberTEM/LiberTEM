import datetime
import time
import os

import numpy as np
import sparse


from libertem.common.sparse import to_dense
from libertem.analysis.gridmatching import calc_coords
from libertem.udf import UDF
import libertem.common.backend as bae
from libertem.udf.raw import PickUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.io.corrections import CorrectionSet
from libertem.io.corrections.detector import correct
from libertem.io.dataset.base.backend import IOBackend, IOBackendImpl


def _naive_mask_apply(masks, data):
    """
    masks: list of masks
    data: 4d array of input data

    returns array of shape (num_masks, scan_y, scan_x)
    """
    assert len(data.shape) == 4
    for mask in masks:
        assert mask.shape == data.shape[2:], "mask doesn't fit frame size"

    dtype = np.result_type(*(m.dtype for m in masks), data.dtype)
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
    coords2 = tuple(np.random.choice(range(c)) for c in size)
    coords10 = tuple(np.random.choice(range(c)) for c in size)
    data[coords2] = np.random.choice(choice) * sum(size)
    data[coords10] = np.random.choice(choice) * 10 * sum(size)
    return data


def _fullgrid(zero, a, b, index, skip_zero=False):
    i, j = np.mgrid[-index:index, -index:index]
    indices = np.concatenate(np.array((i, j)).T)
    if skip_zero:
        select = (np.not_equal(indices[:, 0], 0) + np.not_equal(indices[:, 1], 0))
        indices = indices[select]
    return calc_coords(zero, a, b, indices)


def assert_msg(msg, msg_type, status='ok'):
    print(time.time(), datetime.datetime.now(), msg, msg_type, status)
    assert msg['status'] == status
    assert msg['messageType'] == msg_type,\
        "expected: {}, is: {}".format(msg_type, msg['messageType'])


class PixelsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        assert frame.shape == (16, 16)
        assert self.results.pixelsum.shape == (1,)
        self.results.pixelsum[:] = np.sum(frame)


class MockFile:
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __repr__(self):
        return "<MockFile: [%d, %d)>" % (self.start_idx, self.end_idx)


class DebugDeviceUDF(UDF):
    def __init__(self, backends=None):
        if backends is None:
            backends = ('cupy', 'numpy')
        super().__init__(backends=backends)

    def get_result_buffers(self):
        return {
            'device_id': self.buffer(kind="single", dtype="object"),
            'on_device': self.buffer(kind="sig", dtype=np.float32, where="device"),
            'device_class': self.buffer(kind="nav", dtype="object"),
            'backend': self.buffer(kind="single", dtype="object"),
        }

    def preprocess(self):
        self.results.device_id[0] = dict()
        self.results.backend[0] = dict()

    def process_partition(self, partition):
        cpu = bae.get_use_cpu()
        cuda = bae.get_use_cuda()
        self.results.device_id[0][self.meta.slice] = {
            "cpu": cpu,
            "cuda": cuda
        }
        self.results.on_device[:] += self.xp.sum(partition, axis=0)
        self.results.device_class[:] = self.meta.device_class
        self.results.backend[0][self.meta.slice] = str(self.xp)
        print(f"meta device_class {self.meta.device_class}")

    def merge(self, dest, src):
        de, sr = dest.device_id[0], src.device_id[0]
        for key, value in sr.items():
            assert key not in de
            de[key] = value

        de, sr = dest.backend[0], src.backend[0]
        for key, value in sr.items():
            assert key not in de
            de[key] = value

        dest.on_device[:] += src.on_device
        dest.device_class[:] = src.device_class

    def get_backends(self):
        return self.params.backends


class ValidationUDF(UDF):
    '''
    UDF to compare a dataset against an array-like
    object using a validation function, by default np.allclose().

    Note that the reference should have a flattened nav and the ROI already
    applied since it is compared directly with meta.slice.get()!

    Only works efficiently on large datasets with an inline executor!
    '''
    def __init__(self,
            reference,
            preferred_dtype=UDF.USE_NATIVE_DTYPE,
            validation_function=np.allclose):
        super().__init__(
            reference=reference,
            preferred_dtype=preferred_dtype,
            validation_function=validation_function
        )

    def get_preferred_input_dtype(self):
        return self.params.preferred_dtype

    def get_result_buffers(self):
        return {
            # Just a buffer to "feel" the av shape
            'nav_shape': self.buffer(kind="nav", dtype="float32"),
        }

    def process_tile(self, tile):
        assert self.params.validation_function(
            self.meta.slice.get(self.params.reference), tile
        )


def dataset_correction_verification(ds, roi, lt_ctx, exclude=None):
    """
    compare correct function w/ corrected pick
    """
    for i in range(1):
        shape = (-1, *tuple(ds.shape.sig))
        uncorr = CorrectionSet()
        data = lt_ctx.run_udf(udf=PickUDF(), dataset=ds, roi=roi, corrections=uncorr)

        gain = np.random.random(ds.shape.sig) + 1
        dark = np.random.random(ds.shape.sig) - 0.5

        if exclude is None:
            exclude = [
                (np.random.randint(0, s), np.random.randint(0, s))
                for s in tuple(ds.shape.sig)
            ]

        exclude_coo = sparse.COO(coords=exclude, data=True, shape=ds.shape.sig)
        corrset = CorrectionSet(dark=dark, gain=gain, excluded_pixels=exclude_coo)

        # This one uses native input data
        pick_res = lt_ctx.run_udf(udf=PickUDF(), dataset=ds, corrections=corrset, roi=roi)
        corrected = correct(
            buffer=data['intensity'].raw_data.reshape(shape),
            dark_image=dark,
            gain_map=gain,
            excluded_pixels=exclude,
            inplace=False
        )

        print("Exclude: ", exclude)

        print(pick_res['intensity'].raw_data.dtype)
        print(corrected.dtype)

        assert np.allclose(
            pick_res['intensity'].raw_data.reshape(shape),
            corrected
        )


def dataset_correction_masks(ds, roi, lt_ctx, exclude=None):
    """
    compare correction via sparse mask multiplication w/ correct function
    """
    for i in range(1):
        shape = (-1, *tuple(ds.shape.sig))
        uncorr = CorrectionSet()
        data = lt_ctx.run_udf(udf=PickUDF(), dataset=ds, roi=roi, corrections=uncorr)

        gain = np.random.random(ds.shape.sig) + 1
        dark = np.random.random(ds.shape.sig) - 0.5

        if exclude is None:
            exclude = [
                (np.random.randint(0, s), np.random.randint(0, s))
                for s in tuple(ds.shape.sig)
            ]

        exclude_coo = sparse.COO(coords=exclude, data=True, shape=ds.shape.sig)
        corrset = CorrectionSet(dark=dark, gain=gain, excluded_pixels=exclude_coo)

        def mask_factory():
            s = tuple(ds.shape.sig)
            return sparse.eye(np.prod(s)).reshape((-1, *s))

        # This one casts to float
        mask_res = lt_ctx.run_udf(
            udf=ApplyMasksUDF(mask_factory),
            dataset=ds,
            corrections=corrset,
            roi=roi,
        )
        # This one uses native input data
        corrected = correct(
            buffer=data['intensity'].raw_data.reshape(shape),
            dark_image=dark,
            gain_map=gain,
            excluded_pixels=exclude,
            inplace=False
        )

        print("Exclude: ", exclude)

        print(mask_res['intensity'].raw_data.dtype)
        print(corrected.dtype)

        assert np.allclose(
            mask_res['intensity'].raw_data.reshape(shape),
            corrected
        )


def get_testdata_path():
    return os.environ.get(
        'TESTDATA_BASE_PATH',
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data')
        )
    )


class FakeBackend(IOBackend, id_="fake"):
    def get_impl(self):
        return FakeBackendImpl()


class FakeBackendImpl(IOBackendImpl):
    def get_tiles(
        self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype, decoder,
        sync_offset, corrections,
    ):
        raise RuntimeError("nothing to see here")
        # to make this a generator, there needs to be a yield statement in
        # the body of the function, even if it is never executed:
        yield
