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
            'buf_nav': self.buffer(kind='sig', dtype=np.float32),
            'buf_single': self.buffer(kind='single', dtype=np.float32, extra_shape=(1,)),
        }

    def get_results(self):
        assert self.meta.valid_nav_mask is not None
        assert self.meta.valid_nav_mask.data.sum() > 0, \
            "get_results is not called with an empty valid nav mask"
        if self.params.debug:
            print("get_results", self.meta.valid_nav_mask.data)
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
            print("merge", self.meta.valid_nav_mask.data)
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
