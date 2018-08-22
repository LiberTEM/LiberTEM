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
        INPUT_T loopbuffer[middle_loop * inputstride];
        /* Process an entire block in one go. The code of the inner loop is long.
           Cache-friendlyness? */
        for (int middle = 0; middle < middle_loop; middle++) {
            const int offset = block*blocksize + middle*work_bytes;
            /* input, output 0, 0 */
            source_index = offset + 0*work_bytes;
            target_index = middle*inputstride + 0;
            chunk = *( CHUNK_T*)(inp + source_index);
            loopbuffer[target_index] = decode_op(chunk, -4, 0xff0) | decode_op(chunk, 12, 0xf);

            /* input, output 0, 1 */
            target_index = middle*inputstride + 1;
            loopbuffer[target_index] = decode_op(chunk, 0, 0xf00) | decode_op(chunk, 16, 0xff);

            
            /* input, output 0, 2 */
            target_index = middle*inputstride + 2;
            out_tmp = decode_op(chunk, 20, 0xff0);

            /* input, output 1, 2 */
            source_index = offset + 1*work_bytes;
            chunk = *( CHUNK_T*)(inp + source_index);
            loopbuffer[target_index] = out_tmp | decode_op(chunk, 4, 0xf);
            
            /* input, output 1, 3 */
            target_index = middle*inputstride + 3;
            loopbuffer[target_index] = decode_op(chunk, -8, 0xf00) | decode_op(chunk, 8, 0xff);
            
            /* input, output 1, 4 */
            target_index = middle*inputstride + 4;
            loopbuffer[target_index] = decode_op(chunk, 12, 0xff0) | decode_op(chunk, 28, 0xf);
            
            /* input, output 1, 5 */
            target_index = middle*inputstride + 5;
            out_tmp = decode_op(chunk, 16, 0xf00);

            /* input, output 2, 5 */
            source_index = offset + 2*work_bytes;
            chunk = *( CHUNK_T*)(inp + source_index);
            loopbuffer[target_index] = out_tmp | decode_op(chunk, 0, 0xff);
            
            /* input, output 2, 6 */
            target_index = middle*inputstride + 6;
            loopbuffer[target_index] = decode_op(chunk, 4, 0xff0) | decode_op(chunk, 20, 0xf);
            
            /* input, output 2, 7 */
            target_index = middle*inputstride + 7;
            loopbuffer[target_index] = decode_op(chunk, 8, 0xf00) | decode_op(chunk, 24, 0xff);
        }
        for (int i = 0; i < middle_loop * inputstride; i++){
            out[block*middle_loop * inputstride + i] = loopbuffer[i];
        }
    }
    

    // TODO handle unaligned rest of input here
}

int main(){
    /* Test if gcd is evaluated at compile time.
       This should become the number 4 in the assembly. */
    return gcd(12, 32);
}