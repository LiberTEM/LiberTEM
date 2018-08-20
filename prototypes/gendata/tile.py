import sys

import numpy as np

from libertem.common.slice import Slice

source = np.memmap(sys.argv[1], dtype="float32").reshape(256, 256, 130, 128)

cropped = source[:, :, :128, :128]

del source  # GC source as soon as possible, should be unused after cropping + dtype conversion

dest_shape = (2*256, 3*256, 128, 128)
dest_slice = Slice(origin=(0, 0, 0, 0), shape=dest_shape)
dest_dtype = sys.argv[3]

dest = np.memmap(sys.argv[2], dtype=sys.argv[3], mode="w+", shape=dest_shape)

cropped = cropped.astype(dest_dtype)

for sub in dest_slice.subslices((256, 256, 128, 128)):
    dest[sub.get()] = cropped
