#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

typedef uint32_t pixel_t;
typedef pixel_t mask_t;
typedef uint64_t result_t;

#define SCAN_SIZE (256*256)
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

inline uint32_t extend_and_mask(mask_t mask, int bitpos, uint32_t value) {
    uint32_t is_set = (mask & (1 << bitpos)) >> bitpos;      // 1 or 0
    // is_set == 0 -> -is_set = 0
    // is_set == 1 -> -is_set = 0xFFFFFFFF
    return (-is_set) & value;
}

inline result_t result_for_mask_item(mask_t mask_item, pixel_t *frame, unsigned int frame_offset) {
    result_t res = 0;
    for(unsigned int u = 0; u < MASK_BIT_PER_ITEM / 8; u++) {
        int v = (MASK_BIT_PER_ITEM / 8)*u;
        pixel_t r0 = extend_and_mask(mask_item, v+0, frame[frame_offset + 0 + v]);
        pixel_t r1 = extend_and_mask(mask_item, v+1, frame[frame_offset + 1 + v]);
        pixel_t r2 = extend_and_mask(mask_item, v+2, frame[frame_offset + 2 + v]);
        pixel_t r3 = extend_and_mask(mask_item, v+3, frame[frame_offset + 3 + v]);
        pixel_t r4 = extend_and_mask(mask_item, v+4, frame[frame_offset + 4 + v]);
        pixel_t r5 = extend_and_mask(mask_item, v+5, frame[frame_offset + 5 + v]);
        pixel_t r6 = extend_and_mask(mask_item, v+6, frame[frame_offset + 6 + v]);
        pixel_t r7 = extend_and_mask(mask_item, v+7, frame[frame_offset + 7 + v]);
        res += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    }
    return res;
}

inline result_t single_frame(mask_t* mask, pixel_t* frame, int num_masks, int detector_size) {
    result_t res = 0;
    for(unsigned int p = 0; p < detector_size / MASK_BIT_PER_ITEM / 1; p++) {
        unsigned int k = 1 * p;
        res += result_for_mask_item(mask[k + 0], frame, (k + 0) * MASK_BIT_PER_ITEM);
        /*res += result_for_mask_item(mask[k + 1], frame, (k + 1) * MASK_BIT_PER_ITEM);
        res += result_for_mask_item(mask[k + 2], frame, (k + 2) * MASK_BIT_PER_ITEM);
        res += result_for_mask_item(mask[k + 3], frame, (k + 3) * MASK_BIT_PER_ITEM);*/
    }
    return res;
}

void apply_mask(mask_t* masks, pixel_t* images, result_t* result,
        int scan_size, int num_masks, int detector_size) {
    //assert(detector_size == DETECTOR_SIZE); // for now, help the optimizer...
    //assert(num_masks == NUM_MASKS);
    //assert(scan_size == SCAN_SIZE);
    //assert(detector_size >= 128*128);
    int i = 0;
    for(int f = 0; f < scan_size; f++) {
        for(int m = 0; m < num_masks; m++) {
            result[i] = single_frame(
                    &masks[(detector_size / MASK_BIT_PER_ITEM)*m],
                    &images[detector_size*f],
                    num_masks, detector_size
                    );
            i++;
        }
    }
}
