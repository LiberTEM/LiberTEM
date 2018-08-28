#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include "harness.h"

const int BLOCK_SIZE = 0x5758 - 40;

int main() {
    char buf[15] = {
        0x01, 0x10, 0x00,
        0x01, 0x10, 0x00,
        0x01, 0x10, 0x00,
        0x01, 0x10, 0x00,
        0x01, 0x10, 0x00,
    };

    INPUT_T out[10];

    for(int i = 0; i < 10; i++) {
        out[i] = 0;
    }

    decode_uint12(15, buf, out);

    for(int i = 0; i < 10; i++) {
        if(out[i] != 1) {
            printf("error at index %d: %u != %u\n", i, out[i], 1);
        }
    }
}

