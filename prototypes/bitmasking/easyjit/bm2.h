#include <stdint.h>

typedef uint32_t pixel_t;
typedef uint64_t mask_t;
typedef uint64_t result_t;

#define SCAN_SIZE (16 * 16)
#define DETECTOR_SIZE (128 * 128)
#define NUM_MASKS 1

#define MASK_BIT_PER_ITEM (sizeof(mask_t) * 8)

// these sizes are in number of items of their respective type
#define RESULT_SIZE (NUM_MASKS * SCAN_SIZE)
#define DATASET_SIZE (SCAN_SIZE * DETECTOR_SIZE)
#define MASKS_SIZE (NUM_MASKS * DETECTOR_SIZE / MASK_BIT_PER_ITEM)

extern void apply_masks(mask_t *masks, pixel_t *images, result_t *result, int scan_size, int num_masks, int detector_size);
extern void kernel_simplified(mask_t *masks, pixel_t *images, result_t *result, int scan_size, int num_masks, int detector_size);
