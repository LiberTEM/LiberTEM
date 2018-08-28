#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include "harness.h"

const int BLOCK_SIZE = 0x5758 - 40;

int main() {
    char buf[BLOCK_SIZE];
    FILE *blockfile = fopen("./single-block.raw", "r");
    if(blockfile == NULL) {
        perror("fopen");
        exit(1);
    }
    size_t num_read = fread(buf, 1, BLOCK_SIZE, blockfile);
    if(num_read != BLOCK_SIZE) {
        perror("fread");
        exit(1);
    }

    INPUT_T out[930 * 16];

    for(int i = 0; i < 930*16; i++) {
        out[i] = 0;
    }

    decode_uint12(BLOCK_SIZE, buf, out);

    printf("decoded %d bytes, len(out) = %d\n", BLOCK_SIZE, 930 * 16);

    for(int i = 0; i < 930 * 16; i++) {
        if(out[i] != EXPECTED_OUTPUT[i]) {
            printf("error at index %d: %u != %u\n", i, out[i], EXPECTED_OUTPUT[i]);
        }
    }
}
