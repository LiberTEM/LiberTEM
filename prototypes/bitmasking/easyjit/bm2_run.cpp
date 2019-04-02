#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "bm2.h"

float clock_seconds()
{
    return clock() / (float)CLOCKS_PER_SEC;
}

int main()
{
    float t0 = clock_seconds();

    // init results:
    result_t *results = (result_t *)malloc(sizeof(result_t) * RESULT_SIZE);
    for (unsigned int i = 0; i < RESULT_SIZE; i++)
    {
        results[i] = 0;
    }

    // init masks
    mask_t *maskbuf = (mask_t *)malloc(sizeof(mask_t) * MASKS_SIZE);
    for (unsigned int i = 0; i < MASKS_SIZE; i++)
    {
        maskbuf[i] = ~(maskbuf[i] ^ maskbuf[i]);
    }

    srand(time(NULL));
    // init source data buffer
    pixel_t *dataset = (pixel_t *)malloc(sizeof(pixel_t) * DATASET_SIZE);
    for (unsigned int i = 0; i < DATASET_SIZE; i++)
    {
        dataset[i] = random();
    }
    {
        float delta = clock_seconds() - t0;
        printf("init took %.3fs\n", delta);
    }

    printf("result size: %ldkB, masks size: %ldkB, data size: %ldMB, MASK_BIT_PER_ITEM: %ld\n",
           (sizeof(result_t) * RESULT_SIZE) / 1024,
           (sizeof(mask_t) * MASKS_SIZE) / 1024,
           (sizeof(pixel_t) * DATASET_SIZE) / 1024 / 1024,
           MASK_BIT_PER_ITEM);

    float t1 = clock_seconds();
    unsigned int reps = (256 * 256) / SCAN_SIZE;
    for (unsigned int k = 0; k < reps; k++)
    {
        //kernel_simplified(maskbuf, dataset, results, SCAN_SIZE, NUM_MASKS, DETECTOR_SIZE);
        apply_masks(maskbuf, dataset, results, SCAN_SIZE, NUM_MASKS, DETECTOR_SIZE);
    }
    float delta = clock_seconds() - t1;
    printf("jit: %.3f (%.2fMB/s)\n", delta, reps * sizeof(pixel_t) * DATASET_SIZE / delta / 1024.0 / 1024.0);

    /*
    {
    float t1 = clock_seconds();
    unsigned int reps = (256*256)/SCAN_SIZE;
    for(unsigned int k = 0; k < reps; k++) {
        kernel_simplified(maskbuf, dataset, results, SCAN_SIZE, NUM_MASKS, DETECTOR_SIZE);
    }
    float delta = clock_seconds() - t1;
    printf("non-jit: %.3f (%.2fMB/s)\n", delta, reps*sizeof(pixel_t)*DATASET_SIZE/delta/1024.0/1024.0);
    }
    */

    // this assumes all-1 masks:
    std::cout << "verifying...";
    for (int j = 0; j < RESULT_SIZE; j++)
    {
        for (int frame = 0; frame < SCAN_SIZE; frame++)
        {
            result_t ref = 0;
            for (int i = 0; i < DETECTOR_SIZE; i++)
            {
                ref += dataset[frame * DETECTOR_SIZE + i];
            }
            ref *= reps;
            for (int maskidx = 0; maskidx < NUM_MASKS; maskidx++)
            {
                result_t res = results[maskidx * SCAN_SIZE + frame];
                if (res != ref)
                {
                    std::cout << "invalid result: " << res << "!=" << ref << "\n";
                    abort();
                }
            }
        }
    }
    std::cout << " ok.\n";

    return 0;
}