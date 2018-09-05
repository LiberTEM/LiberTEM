#include <cstring>
#include <cinttypes>
#include <cstdlib>

extern "C" void decode_uint12_impl(uint8_t *inp, uint16_t *out, int size_in, int size_out);

int main() {
    const int size_in = 0x5758 - 40;
    const int size_out = size_in * 8 / 12 * sizeof(uint16_t);
    static_assert(size_out == 930 * 16 * sizeof(uint16_t), "hmmm...");
    uint8_t *input = (uint8_t*)malloc(size_in);

    for(int i = 0; i < size_in; i += 3) {
        input[i+0] = 0xAA;
        input[i+1] = 0xBA;
        input[i+2] = 0xBB;
    }

    uint16_t *output = (uint16_t*)malloc(size_out);

    for(int i = 0; i < 100000; i++) {
        decode_uint12_impl(input, output, size_in, size_out);
    }

    free(input);
    free(output);
}
