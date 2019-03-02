#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <iostream>

typedef unsigned char mask_t;  // each byte storing 8 mask pixels
typedef float pixel_t;
typedef float result_t;

#define SCAN_SIZE (2*4)
#define DETECTOR_SIZE (128*128)
#define NUM_MASKS 3

#define RESULT_SIZE (NUM_MASKS * SCAN_SIZE)
#define DATASET_SIZE (SCAN_SIZE * DETECTOR_SIZE)
#define MASKS_SIZE (NUM_MASKS * DETECTOR_SIZE / 8)

float clock_seconds()
{
    return clock() / (float)CLOCKS_PER_SEC;
}

template <class PixelType>
void apply_mask(mask_t* masks, PixelType* images, PixelType* result) {
    // naive version:
    //
    // for frame in frames:
    //      for mask in masks:
    //          for char in mask:
    //              ...

    int i = 0;
    for(int m = 0; m < NUM_MASKS; m++) {
        for(int f = 0; f < SCAN_SIZE; f++) {
            PixelType res = 0;
            for(int p = 0; p < DETECTOR_SIZE / 8; p++) {
                mask_t maskbyte = masks[(DETECTOR_SIZE/8)*m + p];
                /*
                for(int k = 0; k < 8; k++) {
                    res += (maskbyte & 1 << (7 - k)) ? images[DETECTOR_SIZE*f + p + k] : 0;
                }
                */
                PixelType r0 = (maskbyte & 0x80) ? images[DETECTOR_SIZE*f + p + 0] : 0;
                PixelType r1 = (maskbyte & 0x40) ? images[DETECTOR_SIZE*f + p + 1] : 0;
                PixelType r2 = (maskbyte & 0x20) ? images[DETECTOR_SIZE*f + p + 2] : 0;
                PixelType r3 = (maskbyte & 0x10) ? images[DETECTOR_SIZE*f + p + 3] : 0;
                PixelType r4 = (maskbyte & 0x08) ? images[DETECTOR_SIZE*f + p + 4] : 0;
                PixelType r5 = (maskbyte & 0x04) ? images[DETECTOR_SIZE*f + p + 5] : 0;
                PixelType r6 = (maskbyte & 0x02) ? images[DETECTOR_SIZE*f + p + 6] : 0;
                PixelType r7 = (maskbyte & 0x01) ? images[DETECTOR_SIZE*f + p + 7] : 0;
                res += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
            }
            result[i] = res;
            i++;
        }
    }
}

int main() {
    // init results:
    result_t *results = (result_t*)malloc(sizeof(result_t) * RESULT_SIZE);
    for(int i = 0; i < RESULT_SIZE; i++) {
        results[i] = 0;
    }

    // init masks
    mask_t *maskbuf = (mask_t*)malloc(sizeof(mask_t) * MASKS_SIZE);
    for(int i = 0; i < MASKS_SIZE; i++) {
        maskbuf[i] = 0xFF;
    }

    // init source data buffer
    pixel_t *dataset  = (pixel_t*)malloc(DATASET_SIZE * sizeof(pixel_t));
    for(int i = 0; i < DATASET_SIZE; i++) {
        dataset[i] = 1;
    }

    float t1 = clock_seconds();
    for(int k = 0; k < (256*256)/SCAN_SIZE; k++) {
        apply_mask(maskbuf, dataset, results);
    }
    float delta = clock_seconds() - t1;

    printf("%.3f\n", delta);
    for(int j = 0; j < RESULT_SIZE; j++) {
        if(results[j] != 16384) {
            std::cout << "invalid result: " << results[j] << "\n";
            abort();
        }
    }

    return 0;
}
