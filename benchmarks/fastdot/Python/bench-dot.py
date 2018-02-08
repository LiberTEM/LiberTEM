#!/usr/bin/env python3

'''
Benchmark for different tilings and loop orders for mask application

In order to make best use of the CPU caches and minimize memory transfers, the frames and masks are split in tiles, and tiles from the same region/offset are bundled in stacks. This benchmark shows the benefit from iterating within stacks that stay within a CPU cache compared to traversing large areas of memory, and it gives hints towards good tile and stack sizes. 
'''

import numpy as np
import time
import sys
import csv

# print(np.__config__.show())

# buffer size in byte
bufsize = 256*1024*1024
dtype_data = np.uint32
dtype_mask = np.int32

# Maximum number of masks, used to determine number of repeats for smaller mask sets
# in order to keep the number of mathematical operations constant.
maxmask = 16
						
def iter_tdot(data, masks, repeats):
	stacks = len(data)
	maskcount = len(masks)
	stackheight = len(data[0])
	# Pre-allocate memory for result to not do that over and over again
	result = np.zeros((stackheight, maskcount))
	for repeat in range(repeats):
		for stack in range(stacks):
			result = np.tensordot(data[stack], masks, (1, 1))

def iter_dot(data, masks, repeats):
	stacks = len(data)
	maskcount = len(masks)
	stackheight = len(data[0])
	# Pre-allocate memory for result to not do that over and over again
	result = np.zeros((stackheight, maskcount))
	for repeat in range(repeats):
		for stack in range(stacks):
			result = np.dot(data[stack], masks.T)
			
def iter_naive(data, masks, repeats):
	stacks = len(data)
	maskcount = len(masks)
	stackheight = len(data[0])
	# Pre-allocate memory for result to not do that over and over again
	result = np.zeros((stackheight, maskcount))
	for repeat in range(repeats):
		for stack in range(stacks):
			for frame in range(stackheight):
				for mask in range(maskcount):
					result[frame, mask] = np.dot(data[stack, frame], masks[mask])

def iter_level2(data, masks, repeats):
	stacks = len(data)
	maskcount = len(masks)
	stackheight = len(data[0])
	# Pre-allocate memory for result to not do that over and over again
	result = np.zeros((stackheight, maskcount))
	for repeat in range(repeats):
		for stack in range(stacks):
			for mask in range(maskcount):
				result[:, mask] = np.dot(data[stack], masks[mask])
											
# Convenience data structure to iterate over the functions
functions = (
	('tdot', iter_tdot),
	('dot', iter_dot),
	('naive', iter_naive),
	('level2', iter_level2)
)											
											
def benchmark(maskcount, framesize, stacks, stackheight, repeats):
	'''
	Measure the time required to execute the six different iteration orders.
	'''
	masks = np.ones((maskcount, framesize), dtype=dtype_mask)
	data = np.ones((stacks, stackheight, framesize), dtype=dtype_data)
	result = []
	# "dry run" to get the data into the cache
	iter_naive(data, masks, 2)
	# Apply the different functions on the data
	for (i, it) in functions:
		hotstart = time.clock()
		it(data, masks, repeats)
		hot = time.clock() - hotstart		
		result.append((i, 0, hot))
	return result

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
	# How many frames to bundle
	for stackheight in (4, 8, 16, 32, 64, 128):
		if stackheight > frames:
			break
		stacks = frames // stackheight
		# 8 kPixel to 1 Mpixel
		for maskcount in (1, 2, 4, 8, 16):
			repeats = maxmask // maskcount
			results = benchmark(maskcount, framesize, stacks, stackheight, repeats)
			for (i, blind, hot) in results:
				writer.writerow((count, bufsize, framesize, stackheight, framesize, maskcount, i, blind, hot, hot-blind))
				count += 1
			sys.stdout.flush()
