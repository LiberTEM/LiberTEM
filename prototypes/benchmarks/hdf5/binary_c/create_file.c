#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "common.h"
#include <unistd.h>

int main(int argc, char **argv)
{
    int fd = open("test.raw", O_CREAT|O_WRONLY, 0600);
    pixel_t buf[128];

    for(int i = 0; i < NUM_ITEMS; i += 128)
    {
        for(int j = 0; j < 128; j++) {
            pixel_t r = random() / (pixel_t)RAND_MAX;
            buf[j] = r;
        }
        write(fd, &buf, sizeof(pixel_t) * 128);
    }
}
