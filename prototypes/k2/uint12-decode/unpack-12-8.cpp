#include <inttypes.h>

#define OUT_T uint16_t

/* An attempt to change the sequence of the simple algorithm
   to read several 3-byte blocks with the same alignment relative
   to 32 bit in a loop.
   
   The hope was that the compiler takes care of endianness and such and can vectorize the loop because
   each iteration has the same alignment relative to 32 bit.
   That didn't work, but might be because the code could still be incorrect so that the loop is not aligned.
   
   clang didn't like the narrowing type casting at all and refused to compile this.
*/
   

struct result_tuple {
    const OUT_T k;
    const OUT_T l;
};

constexpr result_tuple decode_3(const char a, const char b, const char c) {
    result_tuple res = {
        /* the narrowing conversion in an initializer tripped clang, or sth like that. */
        a | static_cast<OUT_T>((b & 0x0F) << 8), 
        static_cast<OUT_T>((b & 0xF0) >> 4) | (c << 4)
    };
    return res;
}

void decode_uint12(const int size, char* __restrict__ inp, OUT_T* __restrict__ out) {
    const int inp_tuple = 3;
    const int magic_number = 8;
    const int stride = 64;
    const int blocksize = magic_number * inp_tuple * stride;
    const int blocks = size / blocksize;
    
    for (int block = 0; block < blocks; block++) {
        for (int s = 0; s < stride; s++) {
            for (int m = 0; m < magic_number; m++) {
                const int offset = inp_tuple*(block*blocksize + m*magic_number + s);
                /* guaranteed to be divisible by 3 */
                const int result_offset = offset * 2 / 3;
                const result_tuple res = decode_3(inp[offset], inp[offset + 1], inp[offset + 2]);
                out[result_offset] = res.k;
                out[result_offset+1] = res.l;
            }
        }
    }
 
    

    // TODO handle unaligned rest of input here
}

int main(){
    return 0;
}