#include <inttypes.h>
#include <string.h>

#define CHUNK_T uint32_t
#define INPUT_T uint32_t

extern void decode_uint12(const int size, char* __restrict__ inp, INPUT_T* __restrict__ out);

extern const INPUT_T EXPECTED_OUTPUT[];
