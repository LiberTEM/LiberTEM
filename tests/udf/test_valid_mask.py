import numpy as np
import pytest

from libertem.api import Context
from libertem.common.shape import Shape
from libertem.udf.base import UDF, MergeAttrMapping
from libertem.io.dataset.memory import MemoryDataSet
from libertem.common.buffers import (
    BufferWrapper, InvalidMaskError, get_inner_slice, get_bbox, get_bbox_slice,
)
from libertem.common.math import count_nonzero, prod


class ValidNavMaskUDF(UDF):
    def __init__(self, debug=True):
        super().__init__(debug=debug)

    def get_result_buffers(self):
        return {
            'buf_sig': self.buffer(kind='sig', dtype=np.float32),
            'buf_nav': self.buffer(kind='nav', dtype=np.float32),
            'buf_single': self.buffer(kind='single', dtype=np.float32, extra_shape=(1,)),
        }

    def get_results(self):
        assert self.meta.get_valid_nav_mask() is not None
        assert self.meta.get_valid_nav_mask().sum() > 0, \
            "get_results is not called with an empty valid nav mask"
        assert len(self.meta.get_valid_nav_mask().shape) == 1, \
            "valid_nav_mask should be flattened"
        if self.meta.roi is not None:
            assert self.meta.get_valid_nav_mask().shape[0] == np.count_nonzero(self.meta.roi), \
                "if a `roi` is given, the valid nav mask should be compressed to it by default"
            full_mask = self.meta.get_valid_nav_mask(full_nav=True)
            assert full_mask.shape[0] == self.meta.dataset_shape.nav.size, \
                "when passing `full_nav=True`, the shape must match the flattened ds shape"
        if self.params.debug:
            print("get_results", self.meta.get_valid_nav_mask())
        results = super().get_results()
        return results

    def process_frame(self, frame):
        assert self.meta.get_valid_nav_mask() is None
        assert self.meta.get_valid_nav_mask(full_nav=True) is None
        self.results.buf_sig += frame
        self.results.buf_nav[:] = frame.sum()
        self.results.buf_single[:] = frame.sum()

    def merge(self, dest, src):
        assert self.meta.get_valid_nav_mask() is not None
        assert not np.allclose(True, self.meta.get_valid_nav_mask()), \
            "valid nav mask should be the already merged positions! can't be all-True"
        if self.params.debug:
            print("merge", self.meta.get_valid_nav_mask())
        dest.buf_sig += src.buf_sig
        dest.buf_single += src.buf_single
        dest.buf_nav[:] = src.buf_nav


def test_valid_nav_mask_available(lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=ValidNavMaskUDF()):
        # TODO: maybe compare damage we got in `get_results` with `res.damage` here?
        pass


@pytest.mark.parametrize("with_roi", [True, False])
def test_valid_nav_mask_delayed(delayed_executor, with_roi: bool):
    ctx = Context(executor=delayed_executor)
    if with_roi:
        roi = np.zeros((16, 16), dtype=bool)
        roi[4:-4, 4:-4] = True
    else:
        roi = None
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    for res in ctx.run_udf_iter(dataset=dataset, udf=ValidNavMaskUDF(), roi=roi):
        res.buffers[0]['buf_nav'].data
        assert np.allclose(
            res.buffers[0]['buf_nav'].valid_mask,
            res.damage,
        )


def test_valid_nav_mask_available_roi(lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    roi = np.zeros((16, 16), dtype=bool)
    roi[4:-4, 4:-4] = True
    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=ValidNavMaskUDF(debug=False), roi=roi):
        print("damage", res.damage.data)
        print("raw damage", res.damage.raw_data)


def test_valid_nav_mask_available_random_roi(lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    roi = np.random.choice([True, False], size=(16, 16))
    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=ValidNavMaskUDF(debug=False), roi=roi):
        print("damage", res.damage.data)
        print("raw damage", res.damage.raw_data)


class AdjustValidMaskUDF(UDF):
    def get_result_buffers(self):
        return {
            'all_valid': self.buffer(kind='sig', dtype=np.float32),
            'all_invalid': self.buffer(kind='sig', dtype=np.float32),
            'keep': self.buffer(kind='nav', dtype=np.float32),
            'nav_with_extra': self.buffer(kind='nav', dtype=np.float32, extra_shape=(2,)),
            'custom_2d': self.buffer(kind='single', dtype=np.float32, extra_shape=(64, 64)),
        }

    def get_results(self):
        custom_mask = np.zeros((64, 64), dtype=bool)
        custom_mask[:, 32:] = True

        return {
            'all_valid': self.with_mask(self.results.all_valid, mask=True),
            'all_invalid': self.with_mask(self.results.all_invalid, mask=False),
            'keep': self.results.keep,
            'nav_with_extra': self.results.nav_with_extra,
            'custom_2d': self.with_mask(self.results.custom_2d, mask=custom_mask),
        }

    def process_frame(self, frame):
        self.results.all_valid += frame
        self.results.all_invalid += frame
        self.results.keep[:] = frame.sum()
        self.results.nav_with_extra[:] = frame.sum()
        self.results.custom_2d[:] = 42

    def merge(self, dest, src):
        dest.all_valid += src.all_valid
        dest.all_invalid += src.all_invalid
        dest.custom_2d += src.custom_2d
        dest.keep[:] = src.keep
        dest.nav_with_extra[:] = src.nav_with_extra


@pytest.mark.parametrize(
    "with_roi", [True, False]
)
@pytest.mark.parametrize(
    "executor", ["inline", "delayed"],
)
def test_adjust_valid_mask(with_roi: bool, executor: str, delayed_executor, lt_ctx):
    """
    Test that we can adjust the valid mask in `get_results`
    """
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    if executor == "inline":
        ctx = lt_ctx
    else:
        ctx = Context(executor=delayed_executor)

    if with_roi:
        roi = np.random.choice([True, False], size=dataset.shape.nav)
    else:
        roi = None

    custom_expected = np.zeros((64, 64), dtype=bool)
    custom_expected[:, 32:] = True

    for res in ctx.run_udf_iter(dataset=dataset, udf=AdjustValidMaskUDF(), roi=roi):
        # invariants that hold for any intermediate results:
        # all-valid result:
        valid_mask = res.buffers[0]['all_valid'].valid_mask
        assert np.allclose(valid_mask, True)
        assert valid_mask.shape == res.buffers[0]['all_valid'].data.shape

        # all-invalid result:
        invalid_mask = res.buffers[0]['all_invalid'].valid_mask
        assert np.allclose(invalid_mask, False)
        assert invalid_mask.shape == res.buffers[0]['all_invalid'].data.shape

        # same as "damage", default for kind='nav' buffers:
        keep_mask = res.buffers[0]['keep'].valid_mask
        assert np.allclose(keep_mask, res.damage.data)

        # nav with extra_shape:
        extra_mask = res.buffers[0]['nav_with_extra'].valid_mask
        assert np.allclose(
            extra_mask,
            res.damage.data.reshape((16, 16, 1)),  # broadcastable to nav+extra
        )

        # custom 2d mask:
        custom_mask = res.buffers[0]['custom_2d'].valid_mask
        assert np.allclose(custom_mask, custom_expected)


class CustomMaskFromParams(UDF):
    def __init__(self, mask):
        super().__init__(mask=mask)

    def get_result_buffers(self) -> dict[str, BufferWrapper]:
        return {
            'custom': self.buffer(kind='single', dtype='float32', extra_shape=(64, 64, 3)),
        }

    def get_results(self):
        return {
            'custom': self.with_mask(np.zeros((64, 64, 3), dtype="float32"), mask=self.params.mask),
        }

    def process_frame(self, frame):
        pass

    def merge(self, dest, src):
        pass


@pytest.mark.parametrize("mask_shape", [
    (32, 32),
    (1, 32),
    (64, 64),  # needs to be (64, 64, 1) to be able to broadcast
    (64, 64, 4),
    (1, 1, 4),
    (1, 1, 1, 1),  # that's too many...
])
def test_custom_mask_invalid_shape(mask_shape, lt_ctx):
    """
    Examples of mask shapes that are incompatible with the 'custom' buffer
    defined above. Make sure these raise an appropriate exception.
    """
    dataset = MemoryDataSet(datashape=[16, 16, 4, 4], num_partitions=4)

    mask = np.zeros(mask_shape, dtype=bool)

    with pytest.raises(InvalidMaskError):
        for res in lt_ctx.run_udf_iter(dataset=dataset, udf=CustomMaskFromParams(mask=mask)):
            pass


@pytest.mark.parametrize("mask_dtype", [
    int,
    "float32",
    "complex64",
    # ...
])
def test_custom_mask_invalid_dtype(mask_dtype, lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 4, 4], num_partitions=4)
    mask = np.zeros((16, 16), dtype=mask_dtype)

    with pytest.raises(InvalidMaskError):
        for res in lt_ctx.run_udf_iter(dataset=dataset, udf=CustomMaskFromParams(mask=mask)):
            pass


@pytest.mark.parametrize("mask_shape", [
    (),
    (1,),
    (1, 1),
    (1, 1, 1),
    (64, 64, 1),
    (64, 64, 3),
])
def test_custom_mask_valid(mask_shape, lt_ctx):
    """
    Examples of mask shapes that should be compatible with the shape of the 'custom'
    buffer defined in the `CustomMaskFromParams` UDF.
    """
    dataset = MemoryDataSet(datashape=[16, 16, 4, 4], num_partitions=4)

    mask = np.zeros(mask_shape, dtype=bool)

    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=CustomMaskFromParams(mask=mask)):
        pass


def test_valid_mask_slice_bounding(lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)

    custom_expected = np.zeros((64, 64), dtype=bool)
    custom_expected[:, 32:] = True

    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=AdjustValidMaskUDF()):
        # invariants that hold for any intermediate results:
        # all-valid result:
        buf = res.buffers[0]['all_valid']
        assert buf.data[buf.valid_slice_bounding].shape == buf.data.shape

        # all-invalid result:
        buf = res.buffers[0]['all_invalid']
        assert prod(buf.data[buf.valid_slice_bounding].shape) == 0

        # same as "damage", default for kind='nav' buffers:
        buf = res.buffers[0]['keep']
        assert prod(buf.data[buf.valid_slice_bounding].shape) >= np.count_nonzero(res.damage.data)

        # custom 2d mask:
        buf = res.buffers[0]['custom_2d']
        assert buf.valid_slice_bounding == np.s_[0:64, 32:64]


def test_masked_data(lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)

    custom_expected = np.zeros((64, 64), dtype=bool)
    custom_expected[:, 32:] = True

    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=AdjustValidMaskUDF()):
        for k, buf in res.buffers[0].items():
            # "checksum" over accessing via data[valid_mask] vs. masked_data
            masked_sum = buf.masked_data.sum()  # a value or the marker `masked`

            # if everything is masked out, the masked sum is just the `masked`
            # marker, which is not equal to zero:
            assert np.sum(buf.data[buf.valid_mask]) == masked_sum\
                or np.allclose(buf.masked_data.mask, True)


def test_raw_masked_data(lt_ctx):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)

    roi = np.random.choice(a=[True, False], size=(16, 16))

    custom_expected = np.zeros((64, 64), dtype=bool)
    custom_expected[:, 32:] = True

    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=AdjustValidMaskUDF(), roi=roi):
        for k, buf in res.buffers[0].items():
            # "checksum" over accessing via data[valid_mask] vs. masked_data
            masked_sum = buf.raw_masked_data.sum()  # a value or the marker `masked`

            # if everything is masked out, the masked sum is just the `masked`
            # marker, which is not equal to zero:
            assert np.sum(buf.data[buf.valid_mask]) == masked_sum \
                or np.allclose(buf.raw_masked_data.mask, True)

            # 'nav' is the interesting case where the `roi` is affecting things:
            if buf.kind == 'nav':
                matched_roi = roi.reshape(roi.shape + (1,) * len(buf.extra_shape))
                matched_roi = np.broadcast_to(matched_roi, buf.valid_mask.shape)
                matched_roi = matched_roi.reshape(roi.shape + buf.extra_shape)
                assert np.count_nonzero(~buf.raw_masked_data.mask) == \
                    np.count_nonzero(matched_roi[buf.valid_mask])

            assert np.sum(buf.raw_data[buf._valid_mask]) == masked_sum\
                or np.allclose(buf.raw_masked_data.mask, True)


def test_get_inner_slice():
    a = np.zeros((16, 16), dtype=bool)
    a[5:7] = 1
    a[8, 8] = 1
    a[-1, -1] = 1
    assert get_inner_slice(a, axis=0) == np.s_[5:7, :]

    b = np.zeros((16, 16, 16), dtype=bool)
    b[5:7] = 1
    b[8, 1] = 1
    b[-1, -1] = 1
    assert get_inner_slice(b, axis=0) == np.s_[5:7, :, :]

    c = np.zeros((16, 16, 16), dtype=bool)
    c[:, 5:7, :] = 1
    assert get_inner_slice(c, axis=1) == np.s_[:, 5:7, :]


@pytest.mark.with_numba
def test_get_bbox():
    a = np.zeros((16, 16), dtype=bool)
    a[6, 6] = 1
    assert get_bbox(a) == (6, 6, 6, 6)

    a = np.zeros((16, 16, 16), dtype=bool)
    a[:, 6, 6] = 1
    assert get_bbox(a) == (0, 15, 6, 6, 6, 6)


def test_get_bbox_slice():
    a = np.zeros((16, 16), dtype=bool)
    a[6, 6] = 1
    assert get_bbox_slice(a) == np.s_[6:7, 6:7]

    a = np.zeros((16, 16, 16), dtype=bool)
    a[:, 6, 6] = 1
    assert get_bbox_slice(a) == np.s_[0:16, 6:7, 6:7]

    a = np.zeros((16, 16, 16), dtype=bool)
    a[:, 6, 6] = 1
    a[:, -1, -1] = 1
    assert get_bbox_slice(a) == np.s_[0:16, 6:16, 6:16]


def test_default_mask_nav_extra_shape():
    buf = BufferWrapper(kind='nav', extra_shape=(1, 2), dtype="float32")
    valid_nav_mask = np.array([1, 1, 0], dtype=bool)
    ds_shape = Shape((3, 1, 16, 16), sig_dims=2)
    nav_mask = buf.make_default_mask(
        valid_nav_mask=valid_nav_mask, dataset_shape=ds_shape, roi=None,
    )
    assert nav_mask.shape == (3, 1, 2)
    assert np.allclose(nav_mask, valid_nav_mask.reshape((3, 1, 1)))


def test_default_mask_nav_extra_shape_with_roi():
    buf = BufferWrapper(kind='nav', extra_shape=(1, 2), dtype="float32")
    valid_nav_mask = np.array([1, 0], dtype=bool)
    roi = np.array([True, False, True])
    ds_shape = Shape((3, 1, 16, 16), sig_dims=2)
    nav_mask = buf.make_default_mask(
        valid_nav_mask=valid_nav_mask, dataset_shape=ds_shape, roi=roi,
    )
    assert nav_mask.shape == (2, 1, 2)
    assert np.allclose(nav_mask, valid_nav_mask.reshape((2, 1, 1)))


def test_default_mask_sig_extra_shape():
    buf = BufferWrapper(kind='sig', extra_shape=(1, 2), dtype="float32")
    valid_nav_mask = np.array([1, 1, 0], dtype=bool)
    ds_shape = Shape((3, 1, 32, 32), sig_dims=2)
    sig_mask = buf.make_default_mask(
        valid_nav_mask=valid_nav_mask, dataset_shape=ds_shape, roi=None,
    )
    assert sig_mask.shape == (32, 32, 1, 2)
    assert np.allclose(sig_mask, 1)


def test_default_mask_single_extra_shape():
    buf = BufferWrapper(kind='single', extra_shape=(1, 2), dtype="float32")
    valid_nav_mask = np.array([1, 1, 0], dtype=bool)
    ds_shape = Shape((3, 1, 32, 32), sig_dims=2)
    sig_mask = buf.make_default_mask(
        valid_nav_mask=valid_nav_mask, dataset_shape=ds_shape, roi=None,
    )
    assert sig_mask.shape == (1, 2)
    assert np.allclose(sig_mask, 1)


class CustomValidMask(UDF):
    def get_result_buffers(self):
        nav_shape = self.meta.dataset_shape.nav

        if self.meta.roi is not None:
            nav_shape = (count_nonzero(self.meta.roi),)

        return {
            'custom_2d': self.buffer(kind='single', dtype=np.float32, extra_shape=nav_shape),
        }

    def process_frame(self, frame):
        self.results.custom_2d[self.meta.coordinates] = np.sum(frame)

    def get_results(self):
        # in this case, we set the valid mask of the kind=single buffer to
        # match what a nav buffer would do.
        valid_nav_mask = self.meta.get_valid_nav_mask()

        return {
            'custom_2d': self.with_mask(
                self.results.custom_2d,
                mask=valid_nav_mask.reshape(self.results.custom_2d.shape)
            ),
        }

    def merge(self, dest: MergeAttrMapping, src: MergeAttrMapping):
        dest.custom_2d += src.custom_2d


@pytest.mark.parametrize(
    "with_roi", [True, False]
)
@pytest.mark.parametrize(
    "executor", ["inline", "delayed"],
)
def test_adjust_valid_mask_extra(with_roi: bool, executor: str, delayed_executor, lt_ctx):
    """
    Test that we can adjust the valid mask in `get_results`
    """
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    if executor == "inline":
        ctx = lt_ctx
    else:
        ctx = Context(executor=delayed_executor)

    if with_roi:
        roi = np.random.choice([True, False], size=dataset.shape.nav)
    else:
        roi = None

    ctx.run_udf(dataset=dataset, udf=CustomValidMask(), roi=roi)
