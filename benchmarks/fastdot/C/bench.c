/*
 * Benchmark for different tilings and loop orders for mask application
 *
 * TODO:
 * - for future optimizations, see also:
 *   https://blog.theincredibleholk.org/blog/2012/12/10/optimizing-dot-product/
 *   https://software.intel.com/en-us/articles/use-intriniscs/
 *   http://ok-cleek.com/blogs/?p=20540
 * - analyze generated assembler code to understand applied optimizations
 *   for different compiler flags
 */


#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

typedef double pixel_t;
typedef double result_t;

// NOTE: pixel_v and result_v are used for experimental "manual" vectorization,
//       see bench7
#define VECTOR_SIZE 4
typedef pixel_t __attribute__ ((vector_size (VECTOR_SIZE * sizeof(pixel_t)))) pixel_v;
typedef result_t __attribute__ ((vector_size (VECTOR_SIZE * sizeof(result_t)))) result_v;

#define BUF_SIZE ((size_t)1024*1024*256)            // bytes
#define PX_PER_BUF (BUF_SIZE / sizeof(pixel_t))     // number of pixels
#define MIN_PX_PER_FRAME (128*128)                  // number of pixels
#define MAX_PX_PER_FRAME (4096*4096)                // number of pixels
#define MAX_NUM_MASKS 16                            // number of masks
#define MAX_NUM_FRAMES (PX_PER_BUF / MIN_PX_PER_FRAME)

float clock_seconds()
{
    return clock() / (float)CLOCKS_PER_SEC;
}

/*
 * framesize: number of pixels per frame
 * objidx: the index of the object (mask/frame)
 *
 * tilesize: number of pixels per tile
 * tile: index of the current tile
 *
 * pixelidx: the pixel we want to read, indexed from the start of the tile
 */
pixel_t get_pixel(pixel_t *buf, int framesize, int objidx, int tilesize, int tile, int pixelidx)
{
    return buf[objidx*framesize + tile*tilesize + pixelidx];
}

/*
 * "manually" vectorized version of get_pixel
 */
pixel_v get_pixel_v(pixel_v *buf, int framesize, int objidx, int tilesize, int tile, int pixelidx)
{
    return buf[objidx*framesize + tile*tilesize + pixelidx];
}

/*
 * buf: pointer to pixel buffer that contains the source frame data (contiguous memory, one frame after another)
 * maskbuf: pointer to pixel buffer that contains the masks (contiguous memory, one mask after another)
 * masks: number of masks (one of 1, 2, 4, 8, 16)
 * framesize: number of pixels per frame (e.g. 128*128=16384, 128^2 to 4096^2)
 * frames: number of frames in the dataset (calculated to fit frames*framesize into the buffer)
 * tilesize: number of pixels per tile (8, 16, 32, ... 1024) * 1024
 * repeats: number of times to repeat the whole operation (masks*repeats should be constant across different runs)
 * stackheight: number of tiles from different frames we process while keeping
 *              the same mask tile
 */
void bench1(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;
    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int tile = 0; tile < tiles; tile++) {
                for(int frame = stack_start; frame < stack_end; frame++) {
                    for(int mask = 0; mask < masks; mask++) {
                        for(int i = 0; i < tilesize; i++) {
                            results[frame] += get_pixel(buf, framesize, frame, tilesize, tile, i) \
                                              * get_pixel(maskbuf, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}

void bench2(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;
    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int tile = 0; tile < tiles; tile++) {
                for(int mask = 0; mask < masks; mask++) {
                    for(int frame = stack_start; frame < stack_end; frame++) {
                        for(int i = 0; i < tilesize; i++) {
                            results[frame] += get_pixel(buf, framesize, frame, tilesize, tile, i) \
                                              * get_pixel(maskbuf, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}

void bench3(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;
    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int frame = stack_start; frame < stack_end; frame++) {
                /*
                 * perf remark: we loop over all tiles in the (almost) inner loop, so we don't have
                 * any gains because of the tiling! starting at some frame size, mask plus frame
                 * don't fit the L3 cache anymore and performance degrades
                 */
                for(int tile = 0; tile < tiles; tile++) {
                    for(int mask = 0; mask < masks; mask++) {
                        for(int i = 0; i < tilesize; i++) {
                            results[frame] += get_pixel(buf, framesize, frame, tilesize, tile, i) \
                                              * get_pixel(maskbuf, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}

void bench4(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;
    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int frame = stack_start; frame < stack_end; frame++) {
                for(int mask = 0; mask < masks; mask++) {
                    /*
                     * perf remark: we loop over all tiles in the inner loop, so we don't have
                     * any gains because of the tiling! (the mask is not kept in cache)
                     */
                    for(int tile = 0; tile < tiles; tile++) {
                        for(int i = 0; i < tilesize; i++) {
                            results[frame] += get_pixel(buf, framesize, frame, tilesize, tile, i) \
                                              * get_pixel(maskbuf, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}

void bench5(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;
    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int mask = 0; mask < masks; mask++) {
                for(int tile = 0; tile < tiles; tile++) {
                    for(int frame = stack_start; frame < stack_end; frame++) {
                        for(int i = 0; i < tilesize; i++) {
                            results[frame] += get_pixel(buf, framesize, frame, tilesize, tile, i) \
                                              * get_pixel(maskbuf, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}

void bench6(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;
    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int mask = 0; mask < masks; mask++) {
                for(int frame = stack_start; frame < stack_end; frame++) {
                    /*
                     * perf remark: we loop over all tiles in the inner loop, so we don't have
                     * any gains because of the tiling! (the mask is not kept in cache)
                     */
                    for(int tile = 0; tile < tiles; tile++) {
                        for(int i = 0; i < tilesize; i++) {
                            results[frame] += get_pixel(buf, framesize, frame, tilesize, tile, i) \
                                              * get_pixel(maskbuf, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}

/*
 * "manual" vectorization experiment, does not perform well yet
 */
void bench7(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int tilesize, int repeats, int stackheight, result_t *results)
{
    int tiles = framesize / tilesize;
    int stacks = frames / stackheight;

    pixel_v *buf_v = (pixel_v*) buf;
    pixel_v *maskbuf_v = (pixel_v*) maskbuf;
    result_v *results_v = (result_v*) results;

    // outer loops:
    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            // inner loops:
            int stack_start = stack * stackheight;
            int stack_end = (stack + 1) * stackheight;
            for(int tile = 0; tile < tiles; tile++) {
                for(int mask = 0; mask < masks; mask++) {
                    for(int frame = stack_start; frame < stack_end; frame++) {
                        for(int i = 0; i < tilesize; i++) {
                            results_v[frame] += get_pixel_v(buf_v, framesize, frame, tilesize, tile, i) \
                                                * get_pixel_v(maskbuf_v, framesize, mask, tilesize, tile, i);
                        }
                    }
                }
            }
        }
    }
}


#define BENCH(N) {\
    float t1 = clock_seconds();\
    bench ## N(buf, maskbuf, masks, framesize, frames, tilesize_k * 1024, repeats, stackheight, results);\
    float delta = clock_seconds() - t1;\
    float throughput = (masks * repeats * BUF_SIZE) / delta / 1024 / 1024;\
    printf("%d,%lu,%d,%d,%d,%d,%d,%d,%.8f,%d,%.8f\n",\
            count, BUF_SIZE, framesize, stackheight, tilesize_k*1024, masks, N, 0, delta, 0, throughput);\
    fflush(stdout);\
    count += 1;\
};

int main(int argc, char **argv)
{
    float t1 = clock_seconds();

    // init results:
    result_t *results = (result_t*)malloc(sizeof(result_t) * MAX_NUM_FRAMES);
    for(int i = 0; i < MAX_NUM_FRAMES; i++) {
        results[i] = 0;
    }

    // init masks
    pixel_t *maskbuf = malloc(sizeof(pixel_t) * MAX_PX_PER_FRAME * MAX_NUM_MASKS);
    for(int i = 0; i < (MAX_PX_PER_FRAME * MAX_NUM_MASKS); i++) {
        maskbuf[i] = 1;
    }

    // init source data buffer
    pixel_t *buf  = malloc(BUF_SIZE);
    for(int i = 0; i < PX_PER_BUF; i++) {
        buf[i] = 1;
    }

    float t2 = clock_seconds();

    printf("count,bufsize,framesize,stackheight,tilesize,maskcount,i,blind,hot,hot-blind\n");

    fprintf(stderr, "init took %.8fs\n", t2 - t1);

    int count = 1;

    for(int masks = 1; masks <= 16; masks *= 2) {
        for(int framesize = 256*256; framesize <= 4096*4096; framesize *= 4) {
            int frames = PX_PER_BUF / framesize;
            for(int tilesize_k = 8; tilesize_k <= 1024 && tilesize_k * 1024 <= framesize; tilesize_k *= 2) {
                for(int stackheight = 4; stackheight <= 128 && stackheight <= frames; stackheight *= 2) {
                    for(int i = 0; i < frames; i++) {
                        results[i] = 0;
                    }
                    int repeats = 16 / masks;  // 16 masks => 1 repeat, 8 masks => 2 repeats etc.

                    BENCH(1);
                    BENCH(2);
                    BENCH(3);
                    BENCH(4);
                    BENCH(5);
                    BENCH(6);
//                    BENCH(7);

                    // check correctness
//                    for(int i = 0; i < frames; i++) {
//                        if(results[i] != 6 * repeats * framesize * masks) {
//                            printf("result=%d, framesize=%d, expected=%d\n", results[i], framesize, repeats*framesize);
//                            abort();
//                        }
//                    }
                }
            }
        }
    }
}

