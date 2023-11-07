import numpy as np
import pytest

from libertem.warnings import UseDiscouragedWarning
from libertem.common.exceptions import UDFException
from libertem.udf.base import UDF
from libertem.common.buffers import BufferWrapper


"""
Rules for buffer declarations, `get_result_buffers` and `get_results`:

1) All buffers are declared in `get_result_buffers`
2) If a buffer is only computed in `get_results`, it should be marked via
   `use='result_only'` so it isn't allocated on workers
3) If a buffer is only used as intermediary result, it should be marked via `use='private'`
4) Not including a buffer in `get_results` means it will either be passed on
   unchanged, or dropped if `use='private'`
5) It's an error to omit an `use='result_only'` buffer in `get_results`
6) It's an error to include a `use='private'` buffer in `get_results`
7) All results are returned from `Context.run_udf` as `BufferWrapper` instances
8) By default, if `get_results` is not implemented, `use='private'` buffers are dropped,
   and others are passed through unchanged
"""


class UndeclaredBufferUDF(UDF):
    def get_result_buffers(self):
        return {
            'buf': self.buffer(kind='nav', dtype=np.float32),
        }

    def process_frame(self, frame):
        pass

    def get_results(self):
        return {
            'buf': self.results.buf,
            'blah': np.zeros((128, 128), dtype=np.float64),
        }


def test_undeclared_buffer_error(lt_ctx, default_raw):
    """
    1) All buffers are declared in `get_result_buffers`
    """
    udf = UndeclaredBufferUDF()
    with pytest.raises(KeyError):
        lt_ctx.run_udf(dataset=default_raw, udf=udf)


class NoAllocateResultsUDF(UDF):
    def get_result_buffers(self):
        return {
            'buf1': self.buffer(kind='sig', dtype=np.float32),
            'buf2': self.buffer(kind='sig', dtype=np.float32, use='result_only'),
        }

    def merge(self, dest, src):
        assert 'buf2' not in dest
        assert 'buf2' not in src
        dest.buf1[:] += src.buf1

    def process_frame(self, frame):
        self.results.buf1[:] += frame
        assert self.results.buf2 is None

    def get_results(self):
        return {
            'buf1': self.results.buf1,
            'buf2': self.results.buf1 + 1,
        }


def test_no_allocate_result(lt_ctx, default_raw):
    """
    2) If a buffer is only computed in `get_results`, it should be marked via
       `use='result_only'` so it isn't allocated on workers
    """
    udf = NoAllocateResultsUDF()
    lt_ctx.run_udf(dataset=default_raw, udf=udf)


class UnifyResultTypesUDF(UDF):
    def get_result_buffers(self):
        return {
            'buf1': self.buffer(kind='sig', dtype=np.float32),
            'buf2': self.buffer(kind='sig', dtype=np.float32, use='result_only'),
        }

    def merge(self, dest, src):
        assert 'buf2' not in dest
        assert 'buf2' not in src
        dest.buf1[:] += src.buf1

    def process_frame(self, frame):
        pass

    def get_results(self):
        return {
            'buf1': self.results.buf1,
            'buf2': self.results.buf1 + 1,
        }


def test_unified_result_types(lt_ctx, default_raw):
    """
    7) All results are returned from `Context.run_udf` as `BufferWrapper` instances
    """
    udf = UnifyResultTypesUDF()
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf)
    assert isinstance(results['buf1'], BufferWrapper)
    assert isinstance(results['buf2'], BufferWrapper)


class AverageUDF(UDF):
    """
    Like SumUDF, but also computes the average
    """
    def get_result_buffers(self):
        return {
            'sum': self.buffer(kind='sig', dtype=np.float32),
            'num_frames': self.buffer(kind='single', dtype=np.uint64),
            'average': self.buffer(kind='sig', dtype=np.float32, use='result_only'),
        }

    def process_frame(self, frame):
        self.results.sum[:] += frame
        self.results.num_frames[:] += 1

    def merge(self, dest, src):
        dest.sum[:] += src.sum
        dest.num_frames[:] += src.num_frames

    def get_results(self):
        avg = self.results.sum / self.results.num_frames
        return {
            'sum': self.results.sum,
            'average': avg,
        }


def test_delayed_buffer_alloc(lt_ctx, default_raw):
    udf = AverageUDF()
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf)
    assert np.allclose(
        results['sum'].data / default_raw.shape.nav.size,
        results['average']
    )
    assert results['average'].dtype.kind == 'f'


def test_delayed_buffer_alloc_roi(lt_ctx, default_raw):
    udf = AverageUDF()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert np.allclose(
        results['sum'].data / np.sum(roi),
        results['average']
    )


class SumSigUDFAndAHalf(UDF):
    def get_result_buffers(self):
        return {
            'sum': self.buffer(kind='nav', dtype=np.float32),
            'sum_half': self.buffer(kind='nav', dtype=np.float32, use='result_only'),
        }

    def process_frame(self, frame):
        self.results.sum[:] = np.sum(frame)

    def get_results(self):
        return {
            'sum': self.results.sum,
            'sum_half': self.results.sum / 2,
        }


def test_get_results_nav_with_roi(lt_ctx, default_raw):
    udf = SumSigUDFAndAHalf()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert np.allclose(
        results['sum'].raw_data / 2,
        results['sum_half'].raw_data
    )
    assert np.allclose(
        results['sum'].data / 2,
        results['sum_half'].data,
        equal_nan=True,
    )


class UsePrivate1Defaults(UDF):
    def get_result_buffers(self):
        return {
            'default': self.buffer(kind='nav', dtype=np.float32),
            'private': self.buffer(kind='nav', dtype=np.float32, use='private'),
        }

    def process_frame(self, frame):
        self.results.default[:] = np.sum(frame)
        self.results.private[:] = np.sum(frame)


def test_use_private_defaults(lt_ctx, default_raw):
    """
    3) If a buffer is only used as intermediary result, it should be marked via `use='private'`
    8) By default, if `get_results` is not implemented, `use='private'` buffers are dropped,
       and others are passed through unchanged
    """
    udf = UsePrivate1Defaults()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert 'default' in results
    assert 'private' not in results


class UsePrivate2ImplicitOmit(UsePrivate1Defaults):
    def get_results(self):
        return {
            'default': self.results.default,
        }


def test_use_private_implicit_omit(lt_ctx, default_raw):
    """
    3) If a buffer is only used as intermediary result, it should be marked via `use='private'`
    4) Not including a buffer in `get_results` means it will either be passed on
       unchanged, or dropped if `use='private'`
    """
    udf = UsePrivate2ImplicitOmit()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert 'default' in results
    assert 'private' not in results


class UsePrivate3DontReturn(UsePrivate1Defaults):
    def get_results(self):
        return {
            'private': np.ones(self.meta.dataset_shape.nav),
        }


def test_use_private_dont_include_in_get_results(lt_ctx, default_raw):
    """
    6) It's an error to include a `use='private'` buffer in `get_results`
    """
    udf = UsePrivate3DontReturn()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    with pytest.raises(UDFException):
        lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)


class BufferVisibilityDefault(UDF):
    def get_result_buffers(self):
        return {
            'default': self.buffer(kind='nav', dtype=np.float32),
            'private': self.buffer(kind='nav', dtype=np.float32, use='private'),
            'result_only': self.buffer(kind='nav', dtype=np.float32, use='result_only'),
        }

    def process_frame(self, frame):
        self.results.default[:] = np.sum(frame)
        self.results.private[:] = np.sum(frame)
        assert self.results.result_only is None

    # NOTE: no get_results implemented -> error because of the result_only buffer!


def test_buffer_visibility_default(lt_ctx, default_raw):
    """
    5) It's an error to omit an `use='result_only'` buffer in `get_results`
    """
    udf = BufferVisibilityDefault()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    with pytest.raises(UDFException) as m:
        lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert m.match(
        "don't know how to set use='result_only'"
        " buffer 'result_only'; please implement `get_results`"
    )


class BufferVisibilityResOnly(BufferVisibilityDefault):
    def get_results(self):
        return {
            'result_only': self.results.default,  # just copy over the 'default' buffer
            # NOTE: other keys omitted, this should pass on the non-private keys
        }


def test_buffer_visibility_with_result_only(lt_ctx, default_raw):
    udf = BufferVisibilityResOnly()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert 'result_only' in results
    assert 'default' in results, 'the use=None buffer should be implicitly included in the result'
    assert 'private' not in results

    for k in results:
        np.array(results[k])
        results[k].raw_data
        results[k].data


class ResultOnlyWithFullShape(UDF):
    def get_result_buffers(self):
        return {
            'default': self.buffer(kind='nav', dtype=np.float32),
            'result_only': self.buffer(kind='nav', dtype=np.float32, use='result_only'),
        }

    def process_frame(self, frame):
        self.results.default[:] = np.sum(frame)

    def get_results(self):
        # we want to embed the results ourselves... does this work?
        resonly = np.zeros(self.meta.dataset_shape.nav, dtype=self.results.default.dtype)
        resonly[self.meta.roi] = self.results.default
        return {
            'result_only': resonly,
        }


def test_get_results_nav_with_roi_full_shape(lt_ctx, default_raw):
    udf = ResultOnlyWithFullShape()
    roi = np.random.choice([True, False], size=default_raw.shape.nav)
    results = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert 'result_only' in results
    assert np.allclose(
        results['default'].data[roi],
        results['result_only'].data[roi]
    )
    assert np.allclose(results['result_only'].data[~roi], 0)


class OldDictMergeAccess(UDF):
    def get_result_buffers(self):
        return {
            'default': self.buffer(kind='nav', dtype=np.float32),
        }

    def process_frame(self, frame):
        self.results.default[:] = np.sum(frame)

    def merge(self, dest, src):
        dest['default'][:] = src['default'][:]


def test_warning_for_dict_access_merge(lt_ctx, default_raw):
    udf = OldDictMergeAccess()
    with pytest.warns(UseDiscouragedWarning):
        lt_ctx.run_udf(dataset=default_raw, udf=udf)


class OldUDFDataDictAccess(UDF):
    def get_result_buffers(self):
        return {
            'default': self.buffer(kind='nav', dtype=np.float32),
        }

    def process_frame(self, frame):
        self.results.default[:] = np.sum(frame)

    def postprocess(self):
        self.results['default'].raw_data


def test_warning_for_dict_access_postprocess(lt_ctx, default_raw):
    udf = OldUDFDataDictAccess()
    with pytest.warns(UseDiscouragedWarning):
        lt_ctx.run_udf(dataset=default_raw, udf=udf)
