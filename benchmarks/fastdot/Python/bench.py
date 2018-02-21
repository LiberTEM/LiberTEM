#!/usr/bin/env python3

'''
Benchmark for different tilings and loop orders for mask application

In order to make best use of the CPU caches and minimize memory transfers, the frames and masks are split in tiles, and tiles from the same region/offset are bundled in stacks. This benchmark shows the benefit from iterating within stacks that stay within a CPU cache compared to traversing large areas of memory, and it gives hints towards good tile and stack sizes. 
'''

import numpy as np
import time
import sys
import csv

# buffer size in byte
bufsize = 256*1024*1024
dtype_data = np.float64
dtype_mask = np.float64

# Maximum number of masks, used to determine number of repeats for smaller mask sets
# in order to keep the number of mathematical operations constant.
maxmask = 16

# Function definitions for different orders to iterate through the data set and masks
# based on the thoughts laid out in https://github.com/LiberTEM/LiberTEM/issues/11
# The six functions correspond to the six permutations of (tile, frame, mask)
def iter1(data, masks, repeats):	
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		# Stacks are always iterated in the outer loop to finish one frame stack after the other.
		# This allows displaying a progressing result from stack to stack.
		for stack in range(stacks):
			for tile in range(tiles):
				for frame in range(stackheight):
					for mask in range(maskcount):
						result[stack*stackheight+frame] += np.inner(data[stack,tile, frame], masks[mask, tile])

# The "blind" versions try to emulate all operations, including access to the array data structures,
# except the inner product ,in order to separate overheads from the actual mathematical operation.						
def iter1_blind(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for tile in range(tiles):
				for frame in range(stackheight):
					for mask in range(maskcount):
						result[stack*stackheight+frame] += len(data[stack,tile, frame]) + len(masks[mask, tile])
						
def iter2(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for tile in range(tiles):
				for mask in range(maskcount):
					for frame in range(stackheight):					
						result[stack*stackheight+frame] += np.inner(data[stack,tile, frame], masks[mask, tile])
						
def iter2_blind(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for tile in range(tiles):
				for mask in range(maskcount):
					for frame in range(stackheight):					
						result[stack*stackheight+frame] += len(data[stack,tile, frame]) + len(masks[mask, tile])

def iter3(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for frame in range(stackheight):
				for tile in range(tiles):
					for mask in range(maskcount):
						result[stack*stackheight+frame] += np.inner(data[stack,tile, frame], masks[mask, tile])
						
def iter3_blind(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for frame in range(stackheight):
				for tile in range(tiles):
					for mask in range(maskcount):
						result[stack*stackheight+frame] += len(data[stack,tile, frame]) + len(masks[mask, tile])
						
def iter4(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for frame in range(stackheight):					
				for mask in range(maskcount):
					for tile in range(tiles):
						result[stack*stackheight+frame] += np.inner(data[stack,tile, frame], masks[mask, tile])

def iter4_blind(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for frame in range(stackheight):					
				for mask in range(maskcount):
					for tile in range(tiles):
						result[stack*stackheight+frame] += len(data[stack,tile, frame]) + len(masks[mask, tile])

def iter5(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for mask in range(maskcount):
				for tile in range(tiles):
					for frame in range(stackheight):
						result[stack*stackheight+frame] += np.inner(data[stack,tile, frame], masks[mask, tile])
						
def iter5_blind(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for mask in range(maskcount):
				for tile in range(tiles):
					for frame in range(stackheight):
						result[stack*stackheight+frame] += len(data[stack,tile, frame]) + len(masks[mask, tile])
						
def iter6(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for mask in range(maskcount):
				for frame in range(stackheight):
					for tile in range(tiles):
						result[stack*stackheight+frame] += np.inner(data[stack,tile, frame], masks[mask, tile])
						
def iter6_blind(data, masks, repeats):
	stacks = len(data)
	tiles = len(data[0])
	stackheight = len(data[0,0])
	maskcount = len(masks)
	result = np.zeros(stacks*stackheight)
	for repeat in range(repeats):
		for stack in range(stacks):
			for mask in range(maskcount):
				for frame in range(stackheight):
					for tile in range(tiles):
						result[stack*stackheight+frame] += len(data[stack,tile, frame]) + len(masks[mask, tile])

# Convenience data structure to iterate over the functions
functions = (
	(1, iter1, iter1_blind),
	(2, iter2, iter2_blind),
	(3, iter3, iter2_blind),
	(4, iter4, iter3_blind),
	(5, iter5, iter4_blind),
	(6, iter6, iter5_blind)
)						
						
def benchmark(maskcount, tiles, tilesize, stacks, stackheight, repeats):
	'''
	Measure the time required to execute the six different iteration orders.
	'''
	masks = np.ones((maskcount, tiles, tilesize), dtype=dtype_mask)
	data = np.ones((stacks, tiles, stackheight, tilesize), dtype=dtype_data)
	result = []
	# "dry run" to get the data into the cache
	iter1(data, masks, 2)
	# Apply the different functions on the data
	for (i, it, it_blind) in functions:
		hotstart = time.clock()
		it(data, masks, repeats)
		hot = time.clock() - hotstart
		
		blindstart = time.clock()
		it_blind(data, masks, repeats)
		blind = time.clock() - blindstart
		
		result.append((i, blind, hot))
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
		for tilesize in (8, 16, 32, 64, 128, 256, 512, 1024):
			tilesize *= 1024
			if tilesize > framesize:
				break;
			tiles = framesize // tilesize
			for maskcount in (1, 2, 4, 8, 16):
				repeats = maxmask // maskcount
				results = benchmark(maskcount, tiles, tilesize, stacks, stackheight, repeats)
				for (i, blind, hot) in results:
					writer.writerow((count, bufsize, framesize, stackheight, tilesize, maskcount, i, blind, hot, hot-blind))
					count += 1
				sys.stdout.flush()
