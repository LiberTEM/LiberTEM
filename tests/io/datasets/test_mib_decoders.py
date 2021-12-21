import pytest
import numpy as np
from numpy.testing import assert_allclose

from libertem.io.dataset.mib import (
    encode_r1,
    encode_r6,
    encode_r12,
    decode_r1_swap,
    decode_r6_swap,
    decode_r12_swap,
)


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
