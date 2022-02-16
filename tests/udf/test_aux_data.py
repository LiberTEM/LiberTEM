import numpy as np
import pytest

from libertem.udf import UDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


class EchoUDF(UDF):
    def get_result_buffers(self):
        print("get_result_buffers", self.params.aux.shape)
        return {
            'echo': self.buffer(
                kind="nav", dtype="float32", extra_shape=(2,)
            ),
            'echo_preprocess': self.buffer(
                kind="nav", dtype="float32", extra_shape=(2,)
            ),
            'echo_postprocess': self.buffer(
                kind="nav", dtype="float32", extra_shape=(2,)
            ),
            'weighted': self.buffer(
                kind="nav", dtype="float32",
            )
        }

    def preprocess(self):
        print("preprocess", self.params.aux.shape)
        self.results.echo_preprocess[:] = self.params.aux

    def process_frame(self, frame):
        print("process_frame", self.params.aux.shape)
        self.results.echo[:] = self.params.aux
        self.results.weighted[:] = np.sum(frame) * self.params.aux[0]

    def postprocess(self):
        print("postprocess", self.params.aux.shape)
        self.results.echo_postprocess[:] = self.params.aux


class EchoTiledUDF(UDF):
    def get_result_buffers(self):
        print("get_result_buffers", self.params.aux.shape)
        return {
            'echo': self.buffer(
                kind="nav", dtype="float32", extra_shape=(2,)
            ),
            'weighted': self.buffer(
                kind="nav", dtype="float32",
            ),
        }

    def process_tile(self, tile):
        print(
            "process_tile",
            self.params.aux.shape,
            tile.shape,
            self.results.echo.shape,
            self.results.weighted.shape
        )
        self.results.echo[:] = self.params.aux
        assert len(self.params.aux2.shape) == 1
        assert len(self.results.weighted.shape) == 1
        w = np.sum(tile, axis=(-1, -2)) * self.params.aux[..., 0]
        self.results.weighted[:] += w


def test_aux_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = EchoUDF.aux_data(
        kind="nav",
        data=_mk_random(size=(16, 16, 2), dtype="float32"),
        dtype="float32", extra_shape=(2,)
    )
    dataset = lt_ctx.load(
        "memory", data=data, tileshape=(7, 16, 16),
        num_partitions=2, sig_dims=2
    )

    echo_udf = EchoUDF(aux=aux_data, other_stuff=object())
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'echo_preprocess' in res
    print(data.shape, res['echo_preprocess'].data.shape)
    assert np.allclose(res['echo_preprocess'].raw_data, aux_data.raw_data)
    assert 'echo' in res
    print(data.shape, res['echo'].data.shape)
    assert np.allclose(res['echo'].raw_data, aux_data.raw_data)
    assert 'echo_postprocess' in res
    print(data.shape, res['echo_postprocess'].data.shape)
    assert np.allclose(res['echo_postprocess'].raw_data, aux_data.raw_data)


def test_aux_roi_dummy(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_input = _mk_random(size=(16, 16, 2), dtype="float32")
    aux_data = EchoUDF.aux_data(
        kind="nav",
        data=aux_input,
        dtype="float32", extra_shape=(2,)
    )
    dataset = lt_ctx.load(
        "memory", data=data, tileshape=(7, 16, 16),
        num_partitions=2, sig_dims=2
    )

    roi = np.ones(dataset.shape.nav, dtype="bool")

    echo_udf = EchoUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf, roi=roi)
    assert 'echo_preprocess' in res
    print(data.shape, res['echo_preprocess'].data.shape)
    assert np.allclose(res['echo_preprocess'].data[roi], aux_input[roi])
    assert 'echo' in res
    print(data.shape, res['echo'].data.shape)
    assert np.allclose(res['echo'].data[roi], aux_input[roi])
    assert 'echo_postprocess' in res
    print(data.shape, res['echo_postprocess'].data.shape)
    assert np.allclose(res['echo_postprocess'].data[roi], aux_input[roi])


def test_aux_roi(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_input = _mk_random(size=(16, 16, 2), dtype="float32")
    aux_data = EchoUDF.aux_data(
        kind="nav",
        data=aux_input,
        dtype="float32", extra_shape=(2,)
    )
    dataset = lt_ctx.load(
        "memory", data=data, tileshape=(7, 16, 16),
        num_partitions=2, sig_dims=2
    )

    roi = _mk_random(size=dataset.shape.nav, dtype="bool")

    echo_udf = EchoUDF(aux=aux_data)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf, roi=roi)
    assert 'echo_preprocess' in res
    print(data.shape, res['echo_preprocess'].data.shape)
    assert np.allclose(res['echo_preprocess'].data[roi], aux_input[roi])
    assert 'echo' in res
    print(data.shape, res['echo'].data.shape)
    assert np.allclose(res['echo'].data[roi], aux_input[roi])
    assert 'echo_postprocess' in res
    print(data.shape, res['echo_postprocess'].data.shape)
    assert np.allclose(res['echo_postprocess'].data[roi], aux_input[roi])


def test_aux_2(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    aux_data = EchoUDF.aux_data(
        kind="nav", dtype="float32", extra_shape=(2,),
        data=_mk_random(size=(16, 16, 2), dtype="float32"),
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


@pytest.mark.parametrize(
    "num_partitions", (2, 4)
)
def test_aux_small(num_partitions, lt_ctx):
    data = _mk_random(size=(2, 2, 16, 16), dtype="float32")
    aux_data = EchoUDF.aux_data(
        kind="nav", dtype="float32", extra_shape=(2, ),
        data=_mk_random(size=(2, 2, 2), dtype="float32"),
    )
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=num_partitions, sig_dims=2)

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


@pytest.mark.parametrize(
    'tileshape', [(2, 16, 16), (1, 16, 16), (1, 9, 16)]
)
def test_aux_tiled(lt_ctx, tileshape):
    data = _mk_random(size=(1, 5, 16, 16), dtype="float32")
    aux_data = EchoTiledUDF.aux_data(
        data=_mk_random(size=(1, 5, 2), dtype="float32"),
        kind="nav", dtype="float32", extra_shape=(2,)
    )
    aux_data_2 = EchoTiledUDF.aux_data(
        data=_mk_random(size=(1, 5), dtype="float32"),
        kind="nav", dtype="float32"
    )
    dataset = MemoryDataSet(data=data, tileshape=tileshape,
                            num_partitions=2, sig_dims=2)

    echo_udf = EchoTiledUDF(aux=aux_data, aux2=aux_data_2)
    res = lt_ctx.run_udf(dataset=dataset, udf=echo_udf)
    assert 'weighted' in res
    print(data.shape, res['weighted'].data.shape)
    assert np.allclose(
        res['weighted'].raw_data,
        np.sum(data, axis=(2, 3)).reshape(-1) * aux_data.raw_data[..., 0]
    )
