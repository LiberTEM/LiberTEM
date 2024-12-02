import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

from libertem.io.dataset.base.decode import decode_swap_2, default_decode
from libertem.io.dataset.mib import (
    encode_r1,
    encode_r6,
    encode_r12,
    encode_u1,
    encode_u2,
    decode_r1_swap,
    decode_r6_swap,
    decode_r12_swap,
)


def encode_roundtrip(encode, decode, bits_per_pixel, dtype, shape=(512, 512)):
    max_value = (1 << bits_per_pixel) - 1
    seed = os.environ.get('RANDOM_SEED', None)
    if seed is not None:
        seed = int(seed)
    gen = np.random.default_rng(seed=seed)
    data = gen.integers(low=0, high=max_value + 1, size=shape, dtype=dtype)
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
    return data, encoded, decoded


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'encode,decode,bits_per_pixel,dtype', [
        (encode_r1, decode_r1_swap, 1, "uint8"),
        (encode_r6, decode_r6_swap, 8, "uint8"),
        (encode_r12, decode_r12_swap, 16, "uint16"),
        (encode_u1,  default_decode, 8, "uint8"),
        (encode_u2,  decode_swap_2, 16, "uint16"),
    ],
)
def test_encode_roundtrip(encode, decode, bits_per_pixel, dtype):
    data, encoded, decoded = encode_roundtrip(
        encode, decode, bits_per_pixel, dtype, shape=(256, 256),
    )
    print(list(hex(i) for i in data[0, :8]))
    print(list(hex(i) for i in encoded[0, :16]))
    print(list(hex(i) for i in decoded[0, :16]))
    assert_allclose(data, decoded)
