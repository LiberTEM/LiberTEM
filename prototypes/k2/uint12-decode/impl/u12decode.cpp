#include <cinttypes>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>


static const int middle_loops = 16;

inline int decoded_start_index(int block, int middle_decoded_words) {
    return block*middle_decoded_words;
}

inline int encoded_start_index(int block, int middle_encoded_words) {
    return block*middle_encoded_words;
}

inline int decoded_loopbuffer_index(int middle, int decoded_stride) {
    return middle*decoded_stride;
}

inline int encoded_loopbuffer_index(int middle, int encoded_stride) {
    return middle*encoded_stride;
}

template<class A>
inline A char_start_index(A block, A encoded_bytes, A middle_encoded_words) {
    return block*middle_encoded_words*encoded_bytes;
}

template<typename A, typename B>
inline void decode_triple(A a, A b, A c, B* k, B* l) {
    *k = (a & 0xff) | ((b << 8) & 0xf00);
    *l = ((b >> 4) & 0xf) | ((c << 4) & 0xff0);
}

template<typename Tout>
void decode_uint12_impl_naive(uint8_t* __restrict__ inp, Tout* __restrict__ out,
        int size_in, int size_out) {
    int i = 0, o = 0;
    // let's partially unroll:
    const int middle = 128;
    for(i = 0; i < size_in - (3*middle+1); i += 3*middle) {
        for(int j = 0; j < middle; j++) {
            int base = 3*j+i;

            out[o] = (inp[base] & 0xff) | ((inp[base+1] << 8) & 0xf00);
            out[o+1] = ((inp[base+1] >> 4) & 0xf) | ((inp[base+2] << 4) & 0xff0);

            o += 2;
        }
    }
    for(; i < size_in; i += 3) {
        int base = i;
        out[o] = (inp[base] & 0xff) | ((inp[base+1] << 8) & 0xf00);
        out[o+1] = ((inp[base+1] >> 4) & 0xf) | ((inp[base+2] << 4) & 0xff0);
        o += 2;
    }
}

// size_in, size_out: bytes
template<typename Tout>
void decode_uint12_impl_t(uint8_t* __restrict__ inp, Tout* __restrict__ out,
        int size_in, int size_out) {
    const int bytebits = 8;
    const int bits = 12;
    int decoded_words = size_out / sizeof(Tout);
    assert((size_in * bytebits <= decoded_words * bits));

    const int encoded_bytes = sizeof(uint32_t);

    // how many words of encoded_type (uint32) an encoded block
    const int encoded_stride = 3;
    // how many words of "bits" bits in an encoded block
    const int decoded_stride = 8;

    const uint16_t mask = 0xFFF; // FIXME: type?

    const int middle_decoded_words = middle_loops * decoded_stride;
    const int middle_encoded_words = middle_loops * encoded_stride;
    const int middle_encoded_bytes = middle_encoded_words * encoded_bytes;

    int blocks = decoded_words / middle_decoded_words;

    int rest = size_in % middle_encoded_bytes;

    //uint8_t* loopbuffer_in = static_cast<uint8_t*>(malloc(middle_encoded_bytes));
    //memset(loopbuffer_in, '\0', middle_encoded_bytes);


    // FIXME: nose deamons?
    //uint32_t* loopbuffer_work = reinterpret_cast<uint32_t*>(loopbuffer_in);
    uint32_t* loopbuffer_work = static_cast<uint32_t*>(malloc(middle_encoded_bytes));

    for(int block = 0; block < blocks; block++) {
        int decoded_start = decoded_start_index(block, middle_decoded_words);
        int encoded_char_start = char_start_index(block, encoded_bytes, middle_encoded_words);
        
        //assert(encoded_char_start <= size_in);
        //assert(encoded_char_start + middle_encoded_bytes <= size_in);

        memcpy(loopbuffer_work, &inp[encoded_char_start], middle_encoded_bytes);

        for(int middle = 0; middle < middle_loops; middle++) {
            int decoded_base = decoded_loopbuffer_index(middle, decoded_stride);
            int encoded_base = encoded_loopbuffer_index(middle, encoded_stride);

            out[decoded_start + decoded_base + 0]  =      loopbuffer_work[encoded_base + 0]        & mask;
            out[decoded_start + decoded_base + 1]  =     (loopbuffer_work[encoded_base + 0] >> 12) & mask;
            out[decoded_start + decoded_base + 2]  = ((  (loopbuffer_work[encoded_base + 0] >> 24) & mask)
                                                      | ((loopbuffer_work[encoded_base + 1] <<  8) & mask));
            out[decoded_start + decoded_base + 3]  =     (loopbuffer_work[encoded_base + 1] >>  4) & mask;
            out[decoded_start + decoded_base + 4]  =     (loopbuffer_work[encoded_base + 1] >> 16) & mask;
            out[decoded_start + decoded_base + 5]  = ((  (loopbuffer_work[encoded_base + 1] >> 28) & mask)
                                                      | ((loopbuffer_work[encoded_base + 2] <<  4) & mask));
            out[decoded_start + decoded_base + 6]  =     (loopbuffer_work[encoded_base + 2] >>  8) & mask;
            out[decoded_start + decoded_base + 7]  =     (loopbuffer_work[encoded_base + 2] >> 20) & mask;
        }
    }
    int decoded_remainder_offset = decoded_start_index(blocks, middle_decoded_words);
    int encoded_remainder_offset = char_start_index(blocks, encoded_bytes, middle_encoded_words);
    if (rest > 2) {
        for(int r = 0; r < rest - 2; r += 3) {
            int a = inp[encoded_remainder_offset + r];
            int b = inp[encoded_remainder_offset + r + 1];
            int c = inp[encoded_remainder_offset + r + 2];
            int k, l;
            decode_triple(a, b, c, &k, &l);
            out[decoded_remainder_offset] = k;
            out[decoded_remainder_offset + 1] = l;
            decoded_remainder_offset += 2;
        }
    }
    int tail = rest % 3;
    if (tail > 0) {
        int a = inp[encoded_remainder_offset + rest - 2];
        int b = 0;
        int c = 0;
        if (tail >= 2) {
            b = inp[encoded_remainder_offset + rest - 1];
        }
        // c is always zero, otherwise we wouldn't have a tail!
        int k, l;
        decode_triple(a, b, c, &k, &l);

        out[decoded_remainder_offset] = k;
        if (tail >= 2) {
            out[decoded_remainder_offset + 1] = l;
        }
    }
    //free(loopbuffer_in);
    free(loopbuffer_work);
}

extern "C" {
    void decode_uint12_impl_uint16(uint8_t *inp, uint16_t *out, int size_in, int size_out) {
        return decode_uint12_impl_t(inp, out, size_in, size_out);
    }

    void decode_uint12_impl_float(uint8_t *inp, float *out, int size_in, int size_out) {
        return decode_uint12_impl_t(inp, out, size_in, size_out);
    }

    void decode_uint12_impl_uint16_naive(uint8_t *inp, uint16_t *out, int size_in, int size_out) {
        return decode_uint12_impl_naive(inp, out, size_in, size_out);
    }

    void decode_uint12_impl_float_naive(uint8_t *inp, float *out, int size_in, int size_out) {
        return decode_uint12_impl_naive(inp, out, size_in, size_out);
    }
} // extern "C"
