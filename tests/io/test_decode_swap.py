import itertools

import pytest
import numpy as np

from libertem.io.dataset.base.decode import (
    default_decode, decode_swap_2, decode_swap_4, decode_swap_8,
    DtypeConversionDecoder,
)


@pytest.mark.with_numba
def test_default_decode():
    inp = np.random.randn(1, 16, 16).astype(np.float32)
    out = np.zeros_like(inp, dtype=np.float64)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    default_decode(
        inp.reshape((-1,)).view(dtype=np.uint8),
        out=out.reshape((1, -1,)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
    assert np.allclose(
        inp.astype(np.float64), out
    )


def hexdump(arr):
    arr = arr.reshape((-1,))
    return " ".join("%02x" % i for i in arr.view(dtype=np.uint8))


@pytest.mark.xfail(reason="not implemented yet")
@pytest.mark.with_numba
def test_decode_swap_4_f32():
    inp = np.random.randn(1, 16, 16).astype(np.float32)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    decode_swap_4(
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
    print(hexdump(inp_swapped[:2]))
    print(hexdump(inp.reshape((-1,))[:2]))
    print(hexdump(out.reshape((-1,))[:2]))
    assert np.allclose(inp, out)


@pytest.mark.xfail(reason="not implemented yet")
@pytest.mark.with_numba
def test_decode_swap_4_f64():
    inp = np.random.randn(1, 16, 16).astype(np.float64)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    decode_swap_8(
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
    print(hexdump(inp_swapped[:2]))
    print(hexdump(inp.reshape((-1,))[:2]))
    print(hexdump(out.reshape((-1,))[:2]))
    assert np.allclose(inp, out)


@pytest.mark.with_numba
def test_decode_swap_2_u16():
    inp = np.random.randint(low=0, high=2**15, size=(1, 16, 16), dtype=np.uint16)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    decode_swap_2(
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
    print(hexdump(inp_swapped[:1]))
    print(hexdump(inp.reshape((-1,))[:1]))
    print(hexdump(out.reshape((-1,))[:1]))
    assert np.allclose(inp, out)


@pytest.mark.with_numba
def test_decode_swap_4_u32():
    inp = np.random.randint(low=0, high=2**31, size=(1, 16, 16), dtype=np.uint32)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    decode_swap_4(
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
    print(hexdump(inp_swapped[:1]))
    print(hexdump(inp.reshape((-1,))[:1]))
    print(hexdump(out.reshape((-1,))[:1]))
    assert np.allclose(inp, out)


@pytest.mark.with_numba
def test_decode_swap_4_u64():
    inp = np.random.randint(low=0, high=2**31, size=(1, 16, 16), dtype=np.uint64)
    out = np.zeros_like(inp, dtype=inp.dtype)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])
    inp_swapped = inp.reshape((-1,)).byteswap()
    decode_swap_8(
        inp=inp_swapped.view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )
    print(hexdump(inp_swapped[:2]))
    print(hexdump(inp.reshape((-1,))[:2]))
    print(hexdump(out.reshape((-1,))[:2]))
    assert np.allclose(inp, out)


@pytest.mark.parametrize(
    'dtypes, in_byteorder, out_byteorder',
    itertools.product(
        [
            (np.uint8, np.uint16),
            (np.uint8, np.uint32),
            (np.uint8, np.uint64),
            (np.uint16, np.uint16),
            (np.uint16, np.uint32),
            (np.uint16, np.uint64),
            (np.uint32, np.uint32),
            (np.uint32, np.uint64),

            (np.uint8, np.float32),
            (np.uint16, np.float32),
            (np.uint32, np.float32),

            (np.uint8, np.float64),
            (np.uint16, np.float64),
            (np.uint32, np.float64),
        ],
        ['<', '>', '='],  # input byteorder
        ['='],  # decoding to non-native byteorder is not supported currently
    ),
)
@pytest.mark.with_numba
def test_default_decoder_from_uint(dtypes: tuple[np.dtype], in_byteorder, out_byteorder):
    in_dtype, out_dtype = dtypes
    decoder = DtypeConversionDecoder()
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)
    in_dtype_full = in_dtype.newbyteorder(in_byteorder)
    out_dtype_full = out_dtype.newbyteorder(out_byteorder)
    decode = decoder.get_decode(native_dtype=in_dtype_full, read_dtype=out_dtype_full)
    inp = np.random.randint(
        low=0,
        high=np.iinfo(in_dtype).max,
        size=(1, 16, 16),
        dtype=in_dtype
    )
    if in_dtype_full != in_dtype:
        inp.byteswap()
    print(in_dtype_full, in_dtype, out_dtype_full, out_dtype)
    out = np.zeros_like(inp, dtype=out_dtype_full)
    rr = np.array([0, 0, inp.nbytes])
    origin = np.array([0, 0, 0])
    shape = np.array([1, 16, 16])
    ds_shape = np.array([256, 16, 16])

    if decoder._need_byteswap(in_dtype_full, out_dtype_full):
        assert decoder.get_native_dtype(in_dtype_full, out_dtype_full) == np.uint8
        swapping = True
    else:
        assert decoder.get_native_dtype(in_dtype_full, out_dtype_full) == in_dtype_full
        swapping = False
    decode(
        inp=inp.reshape((-1,)).view(dtype=np.uint8),
        out=out.reshape((1, -1)),
        idx=0,
        native_dtype=inp.dtype,
        rr=rr,
        origin=origin,
        shape=shape,
        ds_shape=ds_shape,
    )

    # swap back and compare:
    if swapping:
        expected = inp.byteswap()
    else:
        expected = inp
    expected = expected.astype(out_dtype_full)

    print(swapping)
    print(hexdump(inp.reshape((-1,))[:1]))
    print(hexdump(expected.reshape((-1,))[:1]))
    print(hexdump(out.reshape((-1,))[:1]))

    assert np.allclose(expected, out)
