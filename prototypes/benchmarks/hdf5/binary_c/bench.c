#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "common.h"

#define FRAME_SIZE 128*128

float clock_seconds()
{
    return clock() / (float)CLOCKS_PER_SEC;
}

#define ITEMS_PER_FRAME (128*128)
#define BUF_FRAMES 8
#define BUF_ITEMS (BUF_FRAMES*ITEMS_PER_FRAME)
#define BUF_SIZE (BUF_ITEMS*sizeof(pixel_t))

pixel_t *init()
{
    pixel_t *buf = (pixel_t*)malloc(BUF_SIZE);
    return buf;
}

result_t result[128*128] __attribute__((used));

int main(int argc, char **argv)
{
    float t0 = clock_seconds();
    FILE *fd = fopen("test.raw", "r");
    pixel_t *buf = init();
    float t1 = clock_seconds();

    int frame_counter = 0;
    for(int j = 0; j < NUM_ITEMS/BUF_ITEMS; j++) {
        fread(buf, sizeof(pixel_t), BUF_ITEMS, fd);
        for(int k = 0; k < BUF_FRAMES; k++) {
            result_t sum = 0;
            for(int i = 0; i < ITEMS_PER_FRAME; i++) {
                sum += buf[i];
            }
            result[frame_counter] = sum;
            frame_counter++;
        }
    }
    
    float delta = clock_seconds() - t0;

    printf("init: %8f\n", t1 - t0);
    printf("%8f\n", delta);
    printf("%2f MB/s\n", FILE_SIZE / delta / 1024 / 1024);
    printf("buffer size: %ld kB\n", BUF_SIZE / 1024);
}
