import math

import pytest
import numpy as np
import numba
from numpy.testing import assert_allclose

from libertem.io.dataset.k2is import decode_uint12_le


def test_foo():
    inp = np.zeros(2, dtype=np.uint16)
    inp[0] = 0xABC
    inp[1] = 0xDEF
    out = np.zeros(int(math.ceil(len(inp) * 12 / 8)), dtype=np.uint8)
    print(list(map(hex, inp)))
    encode_uint12_alt(inp, out)
    print(list(map(hex, out)))
    assert_allclose(out, np.array([0xbc, 0xfa, 0xde]))


@numba.njit
def decode_uint12_le_ref(inp, out):
    """
    Decode bytes from bytestring ``inp`` as 12 bit into ``out``

    The mapping looks like this (where a_i and b_i are nibbles):

    Input (12bit "packed" integers)
    |a_2|a_3|b_3|a_1|b_1|b_2|

    Output (16bit integers):
    |0_0|a_1|a_2|a_3|0_0|b_1|b_2|b_3|

    """
    o = 0
    for i in range(0, len(inp), 3):
        s = inp[i:i + 3]
        a = s[0] | (s[1] & 0x0F) << 8
        b = (s[1] & 0xF0) >> 4 | s[2] << 4
        out[o] = a
        out[o + 1] = b
        o += 2


def encode_uint12_alt(inp, out):
    """
    Encode 12bit values from uint16 `inp` into a uint8 array `out`.

    Input (16bit integers):
    |0_0|a_1|a_2|a_3|0_0|b_1|b_2|b_3|

    Output (12bit "packed" integers)
    |a_2|a_3|b_3|a_1|b_1|b_2|
    """
    o = 0
    for i in range(0, len(inp), 2):
        # two input values, truncated to 12 bit:
        a = inp[i] & 0xFFF
        b = inp[i + 1] & 0xFFF

        j = o
        out[j] = a & 0x0FF
        out[j + 1] = (a & 0xF00) >> 8 | ((b & 0x00F) << 4)
        out[j + 2] = (b & 0xFF0) >> 4

        o += 3


@pytest.mark.with_numba
def test_encode_decode_uint12_ref():
    mult = 45
    inp = np.arange(2*mult, dtype=np.uint16)
    out = np.zeros(int(math.ceil(len(inp) * 12 / 8)), dtype=np.uint8)
    result = np.zeros(int(math.ceil(len(out) * 2 / 3)), dtype=np.uint16)

    print(list(map(hex, inp)))
    encode_uint12_alt(inp, out)

    print(list(map(hex, out.view(np.uint8))))
    decode_uint12_le_ref(inp=out, out=result)

    assert_allclose(inp, result)


@pytest.mark.with_numba
def test_encode_decode_uint12():
    mult = 45
    inp = np.arange(2*mult, dtype=np.uint16)
    out = np.zeros(int(math.ceil(len(inp) * 12 / 8)), dtype=np.uint8)
    result = np.zeros(int(math.ceil(len(out) * 2 / 3)), dtype=np.uint16)
    result_ref = np.zeros(int(math.ceil(len(out) * 2 / 3)), dtype=np.uint16)

    print(list(map(hex, inp)))

    encode_uint12_alt(inp, out)
    print(list(map(hex, out.view(np.uint8))))

    decode_uint12_le(inp=out, out=result)
    assert_allclose(inp, result)

    decode_uint12_le_ref(inp=out, out=result_ref)
    assert_allclose(result_ref, result)
