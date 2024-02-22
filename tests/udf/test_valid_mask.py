import numpy as np

from libertem.api import Context
from libertem.udf.base import UDF
from libertem.io.dataset.memory import MemoryDataSet


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
        assert self.meta.valid_nav_mask is not None
        assert self.meta.valid_nav_mask.sum() > 0, \
            "get_results is not called with an empty valid nav mask"
        if self.params.debug:
            print("get_results", self.meta.valid_nav_mask)
        results = super().get_results()
        # import pdb; pdb.set_trace()
        return results

    def process_frame(self, frame):
        self.results.buf_sig += frame
        self.results.buf_nav[:] = frame.sum()
        self.results.buf_single[:] = frame.sum()

    def merge(self, dest, src):
        assert self.meta.valid_nav_mask is not None
        if self.params.debug:
            print("merge", self.meta.valid_nav_mask)
        dest.buf_sig += src.buf_sig
        dest.buf_single += src.buf_single
        dest.buf_nav[:] = src.buf_nav


def test_valid_nav_mask_available():
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    ctx = Context.make_with('inline')
    for res in ctx.run_udf_iter(dataset=dataset, udf=ValidNavMaskUDF()):
        # TODO: maybe compare damage we got in `get_results` with `res.damage` here?
        pass


def test_valid_nav_mask_available_roi():
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    ctx = Context.make_with('inline')
    roi = np.zeros((16, 16), dtype=bool)
    roi[4:-4, 4:-4] = True
    for res in ctx.run_udf_iter(dataset=dataset, udf=ValidNavMaskUDF(debug=False), roi=roi):
        print("damage", res.damage.data)
        print("raw damage", res.damage.raw_data)


class AdjustValidMaskUDF(UDF):
    def get_result_buffers(self):
        return {
            'all_valid': self.buffer(kind='sig', dtype=np.float32),
            'all_invalid': self.buffer(kind='sig', dtype=np.float32),
            'keep': self.buffer(kind='nav', dtype=np.float32),
            'custom_2d': self.buffer(kind='single', dtype=np.float32, extra_shape=(64, 64)),
        }

    def get_results(self):
        custom_mask = np.zeros((64, 64), dtype=bool)
        custom_mask[:, 32:] = True

        return {
            'all_valid': self.with_mask(self.results.all_valid, mask=1),
            'all_invalid': self.with_mask(self.results.all_invalid, mask=0),
            'keep': self.results.keep,
            'custom_2d': self.with_mask(self.results.custom_2d, mask=custom_mask),
        }

    def process_frame(self, frame):
        self.results.all_valid += frame
        self.results.all_invalid += frame
        self.results.keep[:] = frame.sum()
        self.results.custom_2d[:] = 42

    def merge(self, dest, src):
        dest.all_valid += src.all_valid
        dest.all_invalid += src.all_invalid
        dest.custom_2d += src.custom_2d
        dest.keep[:] = src.keep


def test_adjust_valid_mask():
    """
    Test that we can adjust the valid mask in `get_results`
    """
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32], num_partitions=4)
    ctx = Context.make_with('inline')

    custom_expected = np.zeros((64, 64), dtype=bool)
    custom_expected[:, 32:] = True

    for res in ctx.run_udf_iter(dataset=dataset, udf=AdjustValidMaskUDF()):
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

        # custom 2d mask:
        custom_mask = res.buffers[0]['custom_2d'].valid_mask
        assert np.allclose(custom_mask, custom_expected)
