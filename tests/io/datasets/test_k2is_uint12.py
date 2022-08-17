import math

import pytest
import numpy as np
import numba

from libertem.io.dataset.k2is import decode_uint12_le


@numba.njit
def decode_uint12_le_ref(inp, out):
    """
    decode bytes from bytestring ``inp`` as 12 bit into ``out``
    """
    o = 0
    for i in range(0, len(inp), 3):
        s = inp[i:i + 3]
        a = s[0] | (s[1] & 0x0F) << 8
        b = (s[1] & 0xF0) >> 4 | s[2] << 4
        out[o] = a
        out[o + 1] = b
        o += 2


def encode_12_little_little(inp, out):
    bytebits = 8
    bits = 12
    out = out.view(np.uint8)
    encoded_type = np.dtype('<u4')
    decoded_words = len(inp)
    assert len(out)*bytebits >= decoded_words*bits

    encoded_bytes = encoded_type.itemsize

    # how many words of encoded_type an encoded block
    encoded_stride = 3
    # how many words of "bits" bits in an encoded block
    decoded_stride = 8

    mask = 0xfff

    # How many blocks are encoded in one loop
    middle_loops = 8
    # How many decoded words processed in one middle loop
    middle_decoded_words = middle_loops * decoded_stride
    middle_encoded_words = middle_loops * encoded_stride
    middle_encoded_bytes = middle_encoded_words * encoded_bytes

    blocks = decoded_words // middle_decoded_words

    rest = decoded_words % middle_decoded_words

    def decoded_start_index(block):
        return block*middle_decoded_words

    def encoded_start_index(block):
        return block*middle_encoded_words

    def decoded_loopbuffer_index(middle):
        return middle*decoded_stride

    def encoded_loopbuffer_index(middle):
        return middle*encoded_stride

    def char_start_index(block):
        return block*middle_encoded_words*encoded_bytes

    def encode_pair(k, m):
        a = np.uint8(k & 0xff)
        b = np.uint8(((k & 0xf00) >> 8) | ((m & 0xf) << 4))
        c = np.uint8((m & 0xff0) >> 4)
        return (a, b, c)

    loopbuffer_in = np.zeros(middle_decoded_words, dtype=inp.dtype)
    loopbuffer_out = np.zeros(middle_encoded_words, dtype=encoded_type)
    loopbuffer_out_chars = loopbuffer_out.view(np.uint8)

    for block in range(blocks):
        decoded_start = decoded_start_index(block)
        for i in range(middle_decoded_words):
            loopbuffer_in[i] = inp[decoded_start + i]

        for middle in range(middle_loops):
            decoded_base = decoded_loopbuffer_index(middle)
            encoded_base = encoded_loopbuffer_index(middle)
            loopbuffer_out[encoded_base + 0] = (
                (loopbuffer_in[decoded_base + 0] & mask)
                | ((loopbuffer_in[decoded_base + 1] & mask) << 12)
                | ((loopbuffer_in[decoded_base + 2] & mask) << 24)
            )

            loopbuffer_out[encoded_base + 1] = (
                ((loopbuffer_in[decoded_base + 2] & mask) >> 8)
                | ((loopbuffer_in[decoded_base + 3] & mask) << 4)
                | ((loopbuffer_in[decoded_base + 4] & mask) << 16)
                | ((loopbuffer_in[decoded_base + 5] & mask) << 28)
            )

            loopbuffer_out[encoded_base + 2] = (
                ((loopbuffer_in[decoded_base + 5] & mask) >> 4)
                | ((loopbuffer_in[decoded_base + 6] & mask) << 8)
                | ((loopbuffer_in[decoded_base + 7] & mask) << 20)
            )
        char_start = char_start_index(block)
        for i in range(middle_encoded_bytes):
            out[char_start + i] = loopbuffer_out_chars[i]
    decoded_remainder_offset = decoded_start_index(blocks)
    encoded_remainder_offset = char_start_index(blocks)
    if (rest > 1):
        for r in range(0, rest - 1, 2):
            k = inp[decoded_remainder_offset + r]
            m = inp[decoded_remainder_offset + r + 1]
            (a, b, c) = encode_pair(k, m)
            out[encoded_remainder_offset] = a
            out[encoded_remainder_offset + 1] = b
            out[encoded_remainder_offset + 2] = c
            encoded_remainder_offset += 3
    if (rest % 2):
        k = inp[decoded_remainder_offset + rest - 1]
        m = 0
        (a, b, c) = encode_pair(k, m)
        out[encoded_remainder_offset] = a
        out[encoded_remainder_offset + 1] = b


@pytest.mark.with_numba
def test_encode_decode_uint12_ref():
    mult = 45
    inp = np.arange(2*mult, dtype=np.uint16)
    out = np.zeros(int(math.ceil(len(inp) * 12 / 8)), dtype=np.uint8)
    result = np.zeros(int(math.ceil(len(out) * 2 / 3)), dtype=np.uint16)

    print(list(map(hex, inp)))
    encode_12_little_little(inp, out)

    print(list(map(hex, out.view(np.uint8))))
    decode_uint12_le_ref(inp=out, out=result)

    assert np.allclose(inp, result)


@pytest.mark.with_numba
def test_encode_decode_uint12():
    mult = 45
    inp = np.arange(2*mult, dtype=np.uint16)
    out = np.zeros(int(math.ceil(len(inp) * 12 / 8)), dtype=np.uint8)
    result = np.zeros(int(math.ceil(len(out) * 2 / 3)), dtype=np.uint16)
    result_ref = np.zeros(int(math.ceil(len(out) * 2 / 3)), dtype=np.uint16)

    print(list(map(hex, inp)))

    encode_12_little_little(inp, out)
    print(list(map(hex, out.view(np.uint8))))

    decode_uint12_le(inp=out, out=result)
    assert np.allclose(inp, result)

    decode_uint12_le_ref(inp=out, out=result_ref)
    assert np.allclose(result_ref, result)
