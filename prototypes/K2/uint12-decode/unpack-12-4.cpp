#include <inttypes.h>
#include <string.h>

#define CHUNK_T uint32_t
#define INPUT_T uint32_t

constexpr int gcd(int a, int b)
{
    int temp = 0;
    while (b != 0)
    {
        temp = a % b;

        a = b;
        b = temp;
    }
    return a;
}

constexpr int max(const int a, const int b) {
    return a>b?a:b;
}

constexpr int gcd_2(const int a, const int b)
{
    int i = 1;
    int ret = 1;
    for (i = 1; i <= max(a, b); i++) {
        if ((a%i == 0) && (b%i == 0)) {
            ret = i;
        }
    }
    return ret;
}

constexpr int shift_little(const int bitindex, const int bytesize, const int wordsize) {
    const int byte = bitindex / bytesize;
    const int rest = bitindex % bytesize;
    return byte*bytesize - rest + bytesize - 1;
}

constexpr int shift_big(const int bitindex, const int bytesize, const int wordsize) {
    return wordsize * bytesize - bitindex - 1;
}

struct op_code {
    const int input_index;
    const int output_index;
    const int net_shift;
    const INPUT_T mask;
};

constexpr op_code op_little(const int bits, const int worksize, const int bytesize, const int bit) {
    const int input_index = bit / bits;
    const int output_index = bit / worksize;
    const int input_shift = shift_big(bit % bits, bits, 1);
    const int output_shift = shift_little(bit % worksize, bytesize, worksize);
    const INPUT_T mask = 1 << input_shift;
    const int net_shift = output_shift - input_shift;
    return op_code{input_index, output_index, net_shift, mask};
    
}

constexpr INPUT_T decode_op(CHUNK_T chunk, const int shift, const INPUT_T mask) {
    chunk /= 2^shift;
    return chunk & mask;
}

void decode_uint12(const int size, char* __restrict__ inp, INPUT_T* __restrict__ out) {
    const int work_bytes = sizeof(CHUNK_T);    
    const int bytesize = 8;
    const int worksize = work_bytes * bytesize;
    const int input_bytes = sizeof(INPUT_T);
    const int bits = 12;
    
    const int g = gcd(bits, worksize);
    
    const int workstride = bits / g;
    const int inputstride = worksize / g;
    
    const int total_bits = worksize * workstride;
    
    // 256 AVX SIMD register bits
    const int middle_loop = 64*256 / worksize;
    const int middlesize = middle_loop * work_bytes;
    
    const int blocksize = input_bytes * middlesize * inputstride;
    const int blocks = size / blocksize;
    
    
    for (int block = 0; block < blocks; block++) {
        CHUNK_T chunk;
        CHUNK_T tmp;
        CHUNK_T out_tmp = 0;
        int source_index;
        int target_index;
        INPUT_T loopbuffer_out[middle_loop * inputstride];
        CHUNK_T loopbuffer_in[middle_loop * workstride];
        for (int i = 0; i < middle_loop * workstride; i++) {
            loopbuffer_in[i] = *( CHUNK_T*)(inp + block*blocksize + i);
        }
        for (int i = 0; i < middle_loop * inputstride; i++) {
            loopbuffer_out[i] = 0;
        }
        for (int middle = 0; middle < middle_loop; middle++) {
            /* general version that shifts each bit separately.
               TODO check if operations are grouped; check efficiency.
               This might actually be faster than grouping... 
               It is definitely prettier! */
            for (int bit = 0; bit < total_bits; bit++) {
                const op_code op = op_little(bits, worksize, bytesize, bit);
                const int target_index = middle*inputstride + op.input_index;
                const int source_index = middle*workstride + op.output_index;
                loopbuffer_out[target_index] |= decode_op(loopbuffer_in[source_index], op.net_shift, op.mask);
            }
        }
        for (int i = 0; i < middle_loop * inputstride; i++){
            out[block*middle_loop * inputstride + i] = loopbuffer_out[i];
        }
    }
    

    // TODO handle unaligned rest of input here
}

int main(){
    return gcd(12, 32);
}