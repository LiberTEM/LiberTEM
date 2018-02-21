#!/usr/bin/env python3

import time
import numpy as np
from scipy.ndimage import center_of_mass
import dask.array as da
import hyperspy.api as hs
from hyperspy._lazy_signals import LazySignal2D
import csv
import sys

def center_of_mass_frame(im):
    y, x = center_of_mass(im)
    return np.array([x, y])
	
def bench(repeats, size, chunks):
	data = da.random.random(size, chunks=chunks)
	s = LazySignal2D(data)

	time0 = time.clock()
	for i in range(repeats):
		s_com0 = s.map(
				function=center_of_mass_frame,
				ragged=False, inplace=False)
		s_com0.compute()
		s_com0 = s_com0.T
	return time.clock() - time0

writer = csv.writer(sys.stdout)
writer.writerow(("dim", "time"))
sys.stdout.flush()
for dim in (8, 16, 32, 64):
	repeats = (64 // dim)**2
	t = bench(repeats, (dim, dim, 256, 256), (1, 1, 256, 256))
    # Debug version with less calculation time
	# t = bench(1, (1, 1, 256, 256), (1, 1, 256, 256))
	writer.writerow((dim, t))
	sys.stdout.flush()
	
	
	
