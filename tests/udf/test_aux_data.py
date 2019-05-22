import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.udf import UDF
from utils import MemoryDataSet, _mk_random


class EchoUDF(UDF):
    def get_result_buffers(self):
        return {
            'echo': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        self.results.echo[:] = self.params.aux


def test_aux_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = BufferWrapper(
        kind="nav", dtype="float32",
    )
    aux_data.set_buffer(
        _mk_random(size=(16, 16), dtype="float32")
    )
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    echo_udf = EchoUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'echo' in res
    print(data.shape, res['echo'].data.shape)
    assert np.allclose(res['echo'].raw_data, aux_data.raw_data)
