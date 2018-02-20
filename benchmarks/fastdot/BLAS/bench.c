/*
 * Benchmark for directly using BLAS dgemm functions for different numbers of masks,
 * frame sizes and frame stack heights.
 */


#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cblas.h>

typedef double pixel_t;
typedef double result_t;

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

double wall_clock_delta(struct timespec *start, struct timespec *finish)
{
    double elapsed;
    elapsed = (finish->tv_sec - start->tv_sec);
    elapsed += (finish->tv_nsec - start->tv_nsec) / 1000000000.0;
    return elapsed;
}

/*
 * buf: pointer to pixel buffer that contains the source frame data (contiguous memory, one frame after another)
 * maskbuf: pointer to pixel buffer that contains the masks (contiguous memory, one mask after another)
 * masks: number of masks (one of 1, 2, 4, 8, 16)
 * framesize: number of pixels per frame (e.g. 128*128=16384, 128^2 to 4096^2)
 * frames: number of frames in the dataset (calculated to fit frames*framesize into the buffer)
 * repeats: number of times to repeat the whole operation (masks*repeats should be constant across different runs)
 * stackheight: number of tiles from different frames we process while keeping
 *              the same mask tile
 */
void bench(pixel_t *buf, pixel_t *maskbuf, int masks, int framesize, int frames,
            int repeats, int stackheight, result_t *results)
{
    int m = stackheight;
    int n = masks;
    int k = framesize;
    pixel_t alpha = 1;
    pixel_t beta = 0;
    // lda, ldb, ldc taken from https://software.intel.com/en-us/mkl-developer-reference-c-cblas_dgemmx
    int lda = k;
    int ldb = k;
    int ldc = n;
    int stacks = frames / stackheight;

    for(int repeat = 0; repeat < repeats; repeat++) {
        for(int stack = 0; stack < stacks; stack++) {
            /*
             * operation: C := alpha*op(A)*op(B) + beta*C,
             *
             * parameters of cblas_dgemm:
             * 
             * layout: CblasRowMajor/CblasColMajor
             *
             * transa: transpose applied to A to form op(A) (CblasNoTrans/CblasTrans/CblasConjTrans)
             * transb: transpose applied to B to form op(B) -||-
             *
             * m: number of rows of op(A) and number of rows of C, m >= 0
             * n: number of cols of op(B) and number of cols of C, n >= 0
             * k: number of cols of op(A) and number of rows of op(B), k >= 0
             *
             * alpha: scalar, see operation above
             *
             * a: array, size depends on layout
             * lda: leading dimension of a -> here: framesize (number of pixels per frame)
             *
             * b: array, size depends on layout
             * ldb: leading dimension of b -> here: also framesize (number of pixels per frame)
             *
             * beta: scalar, see operation above
             *
             * c: array, size depends on layout. if beta == 0, c doesn't have to be initialized
             * ldc: leading dimension of c -> here: number of masks
             */

            double *a = &(buf[stack * stackheight]);
            double *b = maskbuf;
            double *c = &(results[stack * stackheight]);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
        }
    }
}

int main(int argc, char **argv)
{
    float t1 = clock_seconds();

    // init results:
    result_t *results = (result_t*)malloc(sizeof(result_t) * MAX_NUM_FRAMES * MAX_NUM_MASKS);
    for(int i = 0; i < MAX_NUM_FRAMES; i++) {
        results[i] = 42;
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

    printf("count,bufsize,framesize,stackheight,maskcount,i,blind,hot,wall_time,hot-blind,throughput\n");

    fprintf(stderr, "init took %.8fs\n", t2 - t1);

    int count = 1;

    for(int masks = 1; masks <= 16; masks *= 2) {
        for(int framesize = 256*256; framesize <= 4096*4096; framesize *= 4) {
            int frames = PX_PER_BUF / framesize;
            for(int stackheight = 4; stackheight <= 128 && stackheight <= frames; stackheight *= 2) {
                for(int i = 0; i < frames; i++) {
                    results[i] = 42;  // canary value to detect if our bounds are correct
                }
                int repeats = 16 / masks;  // 16 masks => 1 repeat, 8 masks => 2 repeats etc.

                {
                    // we use both clock_gettime(CLOCK_MONOTONIC, ...) and clock()
                    // to measure the difference between the real time spent and the CPU time
                    // (clock() includes CPU time for each thread running in this process)
                    struct timespec start, finish;
                    clock_gettime(CLOCK_MONOTONIC, &start);
                    float t1 = clock_seconds();
                    bench(buf, maskbuf, masks, framesize, frames, repeats, stackheight, results);
                    float delta = clock_seconds() - t1;
                    clock_gettime(CLOCK_MONOTONIC, &finish);
                    float wall_time = wall_clock_delta(&start, &finish);
                    float throughput = (masks * repeats * BUF_SIZE) / wall_time / 1024 / 1024;
                    printf("%d,%lu,%d,%d,%d,%d,%d,%.8f,%.8f,%d,%.8f\n",
                            count, BUF_SIZE, framesize, stackheight, masks, 1, 0, delta, wall_time, 0, throughput);
                    fflush(stdout);
                    count += 1;
                }

                for(int i = 0; i < frames; i++) {
                    // input frames and masks are all 1s, so each result should equal the framesize
                    if(results[i] != framesize) {
                        printf("frame %d of %d: result=%f, expected=%d\n", i, frames, results[i], framesize);
                        abort();
                    }
                }
            }
        }
    }
}

