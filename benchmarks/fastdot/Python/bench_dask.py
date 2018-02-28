#!/usr/bin/env python3

'''
Benchmark for different tilings and loop orders for mask application

In order to make best use of the CPU caches and minimize memory transfers, the frames and masks are split in tiles, and tiles from the same region/offset are bundled in stacks. This benchmark shows the benefit from iterating within stacks that stay within a CPU cache compared to traversing large areas of memory, and it gives hints towards good tile and stack sizes. 
'''

import dask.array as da
import dask
import numpy as np
import multiprocessing
import time
import sys
import csv

# print(np.__config__.show())

# buffer size in byte
bufsize = 256*1024*1024
dtype_data = np.float64
dtype_mask = np.float64
dtype_result = np.float64

# Most systems today have 2x hyperthreading
# Hyperthreading is detrimental for numerical codes with optimized CPU cache use
# For that reason we use only half the possible workers.
WORKERS = multiprocessing.cpu_count() // 2

# Maximum number of masks, used to determine number of repeats for smaller mask sets
# in order to keep the number of mathematical operations constant.
maxmask = 16
                        
def iter_tdot(data, masks, repeats):
    frames = len(data)
    maskcount = len(masks)
    # can't preallocate here as tensordot doesn't have an out-parameter
    for repeat in range(repeats):
        result = da.tensordot(data, masks, (1, 1))
        result.compute(num_workers=WORKERS)

def iter_dot(data, masks, repeats):
    frames = len(data)
    maskcount = len(masks)
    for repeat in range(repeats):
        result = da.dot(data, masks.T)
        result.compute(num_workers=WORKERS)
                                            
# Convenience data structure to iterate over the functions
functions = (
    ('tdot', iter_tdot),
    ('dot', iter_dot),
)                                           
                                            
def benchmark(maskcount, framesize, tilesize, stacks, stackheight, repeats):
    '''
    Measure the time required to execute the six different iteration orders.
    '''
    masks = da.ones((maskcount, framesize), dtype=dtype_mask, chunks=(maskcount, tilesize))
    data = da.ones((stacks*stackheight, framesize), dtype=dtype_data, chunks=(stackheight, tilesize))
    result = []
    # "dry run" to get the data into the cache
    iter_dot(data, masks, 2)
    # Apply the different functions on the data
    for (i, it) in functions:
        hotstart = time.clock()
        it(data, masks, repeats)
        hot = time.clock() - hotstart       
        result.append((i, 0, hot))
    return result

def main():
    # global counter as line index in CSV file
    count = 1

    writer = csv.writer(sys.stdout)

    # Headers for CSV file
    writer.writerow(("count", "bufsize", "framesize", "stackheight", "tilesize", "maskcount", "i", "blind", "hot", "hot-blind"))
    sys.stdout.flush()

    # The benchmark expects all sizes, counts etc to be powers of 2

    # Realistic range of frame sizes in px. Maximum corresponds to Gatan K2
    for framesize in (256**2, 512**2, 1024**2, 2048**2, 4096**2):
        # number of frames
        frames = bufsize // framesize // dtype_data(1).itemsize
        for tilesize in (128**2, 256**2, 512**2):
            if tilesize > framesize:
                break
            # How many frames to bundle
            for stackheight in (4, 8, 16, 32, 64, 128):
                if stackheight > frames:
                    break
                stacks = frames // stackheight
                # 8 kPixel to 1 Mpixel
                for maskcount in (1, 2, 4, 8, 16):
                    repeats = maxmask // maskcount
                    results = benchmark(maskcount, framesize, tilesize, stacks, stackheight, repeats)
                    for (i, blind, hot) in results:
                        writer.writerow((count, bufsize, framesize, stackheight, tilesize, maskcount, i, blind, hot, hot-blind))
                        count += 1
                    sys.stdout.flush()

if __name__ == "__main__":
    main()
