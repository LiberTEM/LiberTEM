import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.udf import UDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


class EchoUDF(UDF):
    def get_result_buffers(self):
        return {
            'echo': self.buffer(
                kind="nav", dtype="float32", extra_shape=(2,)
            ),
            'weighted': self.buffer(
                kind="nav", dtype="float32",
            )
        }

    def process_frame(self, frame):
        self.results.echo[:] = self.params.aux
        self.results.weighted[:] = np.sum(frame) * self.params.aux[0]


class EchoTiledUDF(UDF):
    def get_result_buffers(self):
        return {
            'echo': self.buffer(
                kind="nav", dtype="float32", extra_shape=(2,)
            ),
            'weighted': self.buffer(
                kind="nav", dtype="float32",
            )
        }

    def process_tile(self, tile, tile_slice):
        self.results.echo[:] = self.params.aux
        w = np.sum(tile, axis=(-1, -2)) * self.params.aux[..., 0]
        self.results.weighted[:] = w


def test_aux_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = BufferWrapper(
        kind="nav", dtype="float32", extra_shape=(2,)
    )
    aux_data.set_buffer(
        _mk_random(size=(16, 16, 2), dtype="float32")
    )
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    echo_udf = EchoUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'echo' in res
    print(data.shape, res['echo'].data.shape)
    assert np.allclose(res['echo'].raw_data, aux_data.raw_data)


def test_aux_2(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = BufferWrapper(
        kind="nav", dtype="float32", extra_shape=(2,)
    )
    aux_data.set_buffer(
        _mk_random(size=(16, 16, 2), dtype="float32")
    )
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    echo_udf = EchoUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'weighted' in res
    print(data.shape, res['weighted'].data.shape)
    assert np.allclose(
        res['weighted'].raw_data,
        np.sum(data, axis=(2, 3)).reshape(-1) * aux_data.raw_data[..., 0]
    )


def test_aux_constructor(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = EchoUDF.aux_data(
        data=_mk_random(size=(16, 16, 2), dtype="float32"),
        kind="nav", dtype="float32", extra_shape=(2,)
    )
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    echo_udf = EchoUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'weighted' in res
    print(data.shape, res['weighted'].data.shape)
    assert np.allclose(
        res['weighted'].raw_data,
        np.sum(data, axis=(2, 3)).reshape(-1) * aux_data.raw_data[..., 0]
    )


def test_aux_tiled(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = EchoTiledUDF.aux_data(
        data=_mk_random(size=(16, 16, 2), dtype="float32"),
        kind="nav", dtype="float32", extra_shape=(2,)
    )
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    echo_udf = EchoTiledUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'weighted' in res
    print(data.shape, res['weighted'].data.shape)
    assert np.allclose(
        res['weighted'].raw_data,
        np.sum(data, axis=(2, 3)).reshape(-1) * aux_data.raw_data[..., 0]
    )
