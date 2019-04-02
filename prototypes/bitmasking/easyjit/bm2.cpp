#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <easy/jit.h>
#include <easy/code_cache.h>

#include "bm2.h"

result_t inline simple_mask(mask_t mask, int bitpos, pixel_t value)
{
    return (mask & (1L << bitpos)) != 0 ? value : 0;
}

result_t inline branchless_mask(mask_t mask, int bitpos, pixel_t value)
{
    mask_t is_set = (mask & (1L << bitpos)) >> bitpos; // 1 or 0
    // is_set == 0 -> -is_set = 0
    // is_set == 1 -> -is_set = 0xFFFFFFFF
    return (-is_set) & value;
}

mask_t inline get_mask_item(mask_t *masks, int detector_size, int maskidx, int first_pixel_idx)
{
    return masks[(maskidx * detector_size / MASK_BIT_PER_ITEM) + (first_pixel_idx / MASK_BIT_PER_ITEM)];
}

void inline store_to_result(result_t *result, int scan_size, int maskidx, int frame, result_t value)
{
    result[maskidx * scan_size + frame] += value;
}

/*
void kernel_L1(mask_t *masks, pixel_t *images, result_t *result, int scan_size, int num_masks, int detector_size) {
    // L1 ~ 32kB
}

void kernel_L2(mask_t *masks, pixel_t *images, result_t *result, int scan_size, int num_masks, int detector_size) {
    // L2 ~ 256kB
}
*/

void kernel_simplified(mask_t *masks, pixel_t *images, result_t *result, int scan_size, int num_masks, int detector_size)
{
    if (detector_size % MASK_BIT_PER_ITEM != 0)
    {
        printf("detector_size not divisible by MASK_BIT_PER_ITEM\n");
        fflush(stdout);
        abort();
    }
    for (int frame = 0; frame < scan_size; frame++)
    {
        for (unsigned int pixel = 0; pixel < detector_size; pixel += MASK_BIT_PER_ITEM)
        {
            for (int maskidx = 0; maskidx < num_masks; maskidx++)
            {
                result_t res = 0;
                mask_t mask_item = get_mask_item(masks, detector_size, maskidx, pixel);
                for (unsigned int bitpos = 0; bitpos < MASK_BIT_PER_ITEM; bitpos++)
                {
                    result_t tmp = simple_mask(mask_item, bitpos, images[detector_size * frame + pixel + bitpos]);
                    res = res + tmp;
                }
                store_to_result(result, scan_size, maskidx, frame, res);
            }
        }
    }
}

void apply_masks(mask_t *masks, pixel_t *images, result_t *result, int scan_size, int num_masks, int detector_size)
{
    using namespace std::placeholders;
    static easy::Cache<> cache;
    easy::options::opt_level(3, 0);
    easy::options::dump_ir("ir");

    auto const &kernel_opt = cache.jit(kernel_simplified, masks, _1, _2, scan_size, num_masks, detector_size);
    kernel_opt(images, result);
}
