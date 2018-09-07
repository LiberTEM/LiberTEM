from cython cimport view
import numpy as np
from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t

cdef extern void decode_uint12_impl_uint16(const uint8_t *inp, uint16_t *out, int size_in, int size_out);
cdef extern void decode_uint12_impl_float(const uint8_t *inp, float *out, int size_in, int size_out);

cdef extern void decode_uint12_impl_uint16_naive(const uint8_t *inp, uint16_t *out, int size_in, int size_out);
cdef extern void decode_uint12_impl_float_naive(const uint8_t *inp, float *out, int size_in, int size_out);

cdef decoded_start_index(block, middle_decoded_words):
    return block*middle_decoded_words

cdef encoded_start_index(block, middle_encoded_words):
    return block*middle_encoded_words

cdef decoded_loopbuffer_index(middle, decoded_stride):
    return middle*decoded_stride

cdef encoded_loopbuffer_index(middle, encoded_stride):
    return middle*encoded_stride

cdef char_start_index(block, encoded_bytes, middle_encoded_words):
    return block*middle_encoded_words*encoded_bytes

cdef decode_triple(a, b, c):
    k = (a & 0xff) | ((b << 8) & 0xf00)
    l = ((b >> 4) & 0xf) | ((c << 4) & 0xff0)
    return (k, l)

cpdef decode_uint12_cpp_uint16(const uint8_t[::view.contiguous] inp,
                               uint16_t[::view.contiguous] out):
    decode_uint12_impl_uint16(
        inp=&inp[0],
        out=&out[0],
        size_in=len(inp),
        size_out=out.nbytes,
    )

cpdef decode_uint12_cpp_float(const uint8_t[::view.contiguous] inp,
                              float[::view.contiguous] out):
    decode_uint12_impl_float(
        inp=&inp[0],
        out=&out[0],
        size_in=len(inp),
        size_out=out.nbytes,
    )

cpdef decode_uint12_cpp_uint16_naive(const uint8_t[::view.contiguous] inp,
                                     uint16_t[::view.contiguous] out):
    decode_uint12_impl_uint16_naive(
        inp=&inp[0],
        out=&out[0],
        size_in=len(inp),
        size_out=out.nbytes,
    )

cpdef decode_uint12_cpp_float_naive(const uint8_t[::view.contiguous] inp,
                                    float[::view.contiguous] out):
    decode_uint12_impl_float_naive(
        inp=&inp[0],
        out=&out[0],
        size_in=len(inp),
        size_out=out.nbytes,
    )



cpdef decode_uint12(const uint8_t[::view.contiguous] inp,
                    uint16_t[::view.contiguous] out):
    cdef uint32_t i
    cdef uint32_t r
    cdef uint32_t encoded_char_start
    cdef uint32_t decoded_start
    cdef uint32_t encoded_remainder_offset
    cdef uint32_t decoded_remainder_offset
    cdef uint32_t rest

    bytebits = 8
    bits = 12
    encoded_type = np.dtype('<u4')
    decoded_words = len(out)
    assert(len(inp)*bytebits <= decoded_words*bits)

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

    rest = len(inp) % middle_encoded_bytes


    loopbuffer_in  = np.zeros(middle_encoded_bytes, dtype=np.uint8)
    loopbuffer_work = loopbuffer_in.view(encoded_type) # dtype here: uint32 little endian
    loopbuffer_out = np.zeros(middle_decoded_words, dtype="uint16")  # dtype here: uint16
    # loopbuffer_out = np.zeros(middle_decoded_words, dtype=out.dtype)  # dtype here: uint16

    for block in range(blocks):
        decoded_start = decoded_start_index(block, middle_decoded_words)
        decoded_stop = decoded_start_index(block + 1, middle_decoded_words)
        encoded_start = encoded_start_index(block, middle_encoded_words)
        encoded_char_start = char_start_index(block, encoded_bytes, middle_encoded_words)

        for i in range(middle_encoded_bytes):
            loopbuffer_in[i] = inp[encoded_char_start + i]

        for middle in range(middle_loops):
            decoded_base = decoded_loopbuffer_index(middle, decoded_stride)
            encoded_base = encoded_loopbuffer_index(middle, encoded_stride)
   
            loopbuffer_out[decoded_base + 0]  =      loopbuffer_work[encoded_base + 0]        & mask
            loopbuffer_out[decoded_base + 1]  =     (loopbuffer_work[encoded_base + 0] >> 12) & mask
            loopbuffer_out[decoded_base + 2]  = ((  (loopbuffer_work[encoded_base + 0] >> 24) & mask)
                                                 | ((loopbuffer_work[encoded_base + 1] <<  8) & mask))
            loopbuffer_out[decoded_base + 3]  =     (loopbuffer_work[encoded_base + 1] >>  4) & mask
            loopbuffer_out[decoded_base + 4]  =     (loopbuffer_work[encoded_base + 1] >> 16) & mask
            loopbuffer_out[decoded_base + 5]  = ((  (loopbuffer_work[encoded_base + 1] >> 28) & mask) 
                                                 | ((loopbuffer_work[encoded_base + 2] <<  4) & mask))
            loopbuffer_out[decoded_base + 6]  =     (loopbuffer_work[encoded_base + 2] >>  8) & mask
            loopbuffer_out[decoded_base + 7]  =     (loopbuffer_work[encoded_base + 2] >> 20) & mask

        for i in range(middle_decoded_words):
            out[decoded_start + i] = loopbuffer_out[i]
    decoded_remainder_offset = decoded_start_index(blocks, middle_decoded_words)
    encoded_remainder_offset = char_start_index(blocks, encoded_bytes, middle_encoded_words)
    if (rest > 2):
        for r in range(0, rest - 2, 3):
            a = inp[encoded_remainder_offset + r]
            b = inp[encoded_remainder_offset + r + 1]
            c = inp[encoded_remainder_offset + r + 2]
            (k, l) = decode_triple(a, b, c)
            out[decoded_remainder_offset] = k
            out[decoded_remainder_offset + 1] = l
            decoded_remainder_offset += 2
    tail = rest % 3
    if tail > 0:
        a = inp[encoded_remainder_offset + rest - 2]
        (b, c) = (0, 0)
        if tail >= 2:
            b = inp[encoded_remainder_offset + rest - 1]
        # c is always zero, otherwise we wouldn't have a tail!
        (k, l) = decode_triple(a, b, c)

        out[decoded_remainder_offset] = k
        if tail >= 2:
            out[decoded_remainder_offset + 1] = l
