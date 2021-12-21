import pytest
import numba
import numpy as np
from numpy.testing import assert_allclose

from libertem.io.dataset.mib import (
    decode_r1_swap,
    decode_r6_swap,
    decode_r12_swap,
)


# These encoders takes 2D input/output data - this means we can use
# strides to do slicing and reversing. 2D input data means one output
# row (of bytes) corresponds to one input row (of pixels).


@numba.njit(cache=True)
def encode_u1(inp, out):
    for y in range(out.shape[0]):
        out[y] = inp[y]


@numba.jit(cache=True)
def encode_u2(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0]):
            in_value = row_in[i]
            row_out[i * 2] = (0xFF00 & in_value) >> 8
            row_out[i * 2 + 1] = 0xFF & in_value


@numba.njit(cache=True)
def encode_r1(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for stripe in range(row_out.shape[0] // 8):
            for byte in range(8):
                out_byte = 0
                for bitpos in range(8):
                    value = row_in[64 * stripe + 8 * byte + bitpos] & 1
                    out_byte |= (value << bitpos)
                row_out[(stripe + 1) * 8 - (byte + 1)] = out_byte


@numba.njit(cache=True)
def encode_r6(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_out.shape[0]):
            col = i % 8
            pos = i // 8
            in_pos = (pos + 1) * 8 - col - 1
            row_out[i] = row_in[in_pos]


@numba.njit(cache=True)
def encode_r12(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0]):
            col = i % 4
            pos = i // 4
            in_pos = (pos + 1) * 4 - col - 1
            in_value = row_in[in_pos]
            row_out[i * 2] = (0xFF00 & in_value) >> 8
            row_out[i * 2 + 1] = 0xFF & in_value


def encode_roundtrip(encode, decode, bits_per_pixel, shape=(512, 512)):
    max_value = (1 << bits_per_pixel) - 1
    data = np.random.randint(0, max_value + 1, shape)
    encoded = np.zeros(data.size // 8 * bits_per_pixel, dtype=np.uint8)
    encoded = encoded.reshape((shape[0], -1))
    encode(inp=data, out=encoded)
    decoded = np.zeros_like(data).reshape((1, -1))
    decode(
        inp=encoded.reshape((-1,)),
        out=decoded,
        idx=0,
        # the parameters below are not used by the mib decoders:
        native_dtype=np.uint8,
        rr=np.zeros(1, dtype=np.uint64),
        origin=np.zeros(1, dtype=np.uint64),
        shape=np.zeros(1, dtype=np.uint64),
        ds_shape=np.zeros(1, dtype=np.uint64),
    )
    decoded = decoded.reshape(data.shape)
    return data, decoded


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'encode,decode,bits_per_pixel', [
        (encode_r1, decode_r1_swap, 1),
        (encode_r6, decode_r6_swap, 8),
        (encode_r12, decode_r12_swap, 16),
    ],
)
def test_encode_roundtrip(encode, decode, bits_per_pixel):
    data, decoded = encode_roundtrip(encode, decode, bits_per_pixel, shape=(256, 256))
    assert_allclose(data, decoded)
