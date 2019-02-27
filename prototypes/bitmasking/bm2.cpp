#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

typedef uint32_t pixel_t;
typedef pixel_t mask_t;
typedef uint64_t result_t;

#define SCAN_SIZE (16*16)
#define DETECTOR_SIZE (128*128)
#define NUM_MASKS 1

#define MASK_BIT_PER_ITEM (sizeof(mask_t) * 8)
#define RESULT_SIZE (NUM_MASKS * SCAN_SIZE)
#define DATASET_SIZE (SCAN_SIZE * DETECTOR_SIZE)
#define MASKS_SIZE (NUM_MASKS * DETECTOR_SIZE / MASK_BIT_PER_ITEM)

float clock_seconds()
{
    return clock() / (float)CLOCKS_PER_SEC;
}

inline uint16_t extendAndMask(mask_t& mask, int bitpos, uint16_t value) {
    uint16_t is_set = (mask & (1 << bitpos)) >> bitpos;      // 1 or 0
    // is_set == 0 -> -is_set = 0
    // is_set == 1 -> -is_set = 0xFFFFFFFF
    return (-is_set) & value;
}

inline uint32_t extendAndMask(mask_t& mask, int bitpos, uint32_t value) {
    uint32_t is_set = (mask & (1 << bitpos)) >> bitpos;      // 1 or 0
    // is_set == 0 -> -is_set = 0
    // is_set == 1 -> -is_set = 0xFFFFFFFF
    return (-is_set) & value;
}

inline uint64_t extendAndMask(mask_t& mask, int bitpos, uint64_t value) {
    uint64_t is_set = (mask & (1 << bitpos)) >> bitpos;      // 1 or 0
    // is_set == 0 -> -is_set = 0
    // is_set == 1 -> -is_set = 0xFFFFFFFF
    return (-is_set) & value;
}

void apply_mask(mask_t* masks, pixel_t* images, result_t* result) {
    int i = 0;
    for(int f = 0; f < SCAN_SIZE; f++) {
        for(int m = 0; m < NUM_MASKS; m++) {
            result_t res = 0;
            for(unsigned int p = 0; p < DETECTOR_SIZE / MASK_BIT_PER_ITEM; p++) {
                mask_t mask_item = masks[(DETECTOR_SIZE / MASK_BIT_PER_ITEM)*m + (p/MASK_BIT_PER_ITEM)];
                for(unsigned int u = 0; u < MASK_BIT_PER_ITEM; u += 8) {
                    {
                        pixel_t r0 = extendAndMask(mask_item, u+0, images[DETECTOR_SIZE*f + p + u+0]);
                        pixel_t r1 = extendAndMask(mask_item, u+1, images[DETECTOR_SIZE*f + p + u+1]);
                        pixel_t r2 = extendAndMask(mask_item, u+2, images[DETECTOR_SIZE*f + p + u+2]);
                        pixel_t r3 = extendAndMask(mask_item, u+3, images[DETECTOR_SIZE*f + p + u+3]);
                        res += r0 + r1 + r2 + r3;
                    }
                    {
                        pixel_t r0 = extendAndMask(mask_item, u+4, images[DETECTOR_SIZE*f + p + u+4]);
                        pixel_t r1 = extendAndMask(mask_item, u+5, images[DETECTOR_SIZE*f + p + u+5]);
                        pixel_t r2 = extendAndMask(mask_item, u+6, images[DETECTOR_SIZE*f + p + u+6]);
                        pixel_t r3 = extendAndMask(mask_item, u+7, images[DETECTOR_SIZE*f + p + u+7]);
                        res += r0 + r1 + r2 + r3;
                    }
                }
            }
            result[i] = res;
            i++;
        }
    }
    if(i != RESULT_SIZE) {
        abort();
    }
}

int main() {
    float t0 = clock_seconds();

    // init results:
    result_t *results = (result_t*)malloc(sizeof(result_t) * RESULT_SIZE);
    for(unsigned int i = 0; i < RESULT_SIZE; i++) {
        results[i] = 0;
    }

    // init masks
    mask_t *maskbuf = (mask_t*)malloc(sizeof(mask_t) * MASKS_SIZE);
    for(unsigned int i = 0; i < MASKS_SIZE; i++) {
        maskbuf[i] = ~(maskbuf[i] ^ maskbuf[i]);
    }

    // init source data buffer
    pixel_t *dataset  = (pixel_t*)malloc(DATASET_SIZE * sizeof(pixel_t));
    for(unsigned int i = 0; i < DATASET_SIZE; i++) {
        dataset[i] = 1;
    }
    {
        float delta = clock_seconds() - t0;
        printf("init took %.3fs\n", delta);
    }

    float t1 = clock_seconds();
    unsigned int reps = (256*256)/SCAN_SIZE;
    for(unsigned int k = 0; k < reps; k++) {
        apply_mask(maskbuf, dataset, results);
    }
    float delta = clock_seconds() - t1;

    printf("%.3f (%.2fMB/s)\n", delta, reps*sizeof(pixel_t)*DATASET_SIZE/delta/1024.0/1024.0);

    for(int j = 0; j < RESULT_SIZE; j++) {
        if(results[j] != DETECTOR_SIZE) {
            std::cout << "invalid result: " << results[j] << "\n";
            abort();
        }
    }

    return 0;
}

