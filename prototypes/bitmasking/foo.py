import numpy as np
import time
import test

t0 = time.time()

num_masks = 1
scan = (256, 256)
detector_shape = (128, 128)
masks = (np.ones((num_masks,) + detector_shape, dtype=np.uint32) * 0xFFFFFFFF).ravel()
images = np.ones(scan + detector_shape, dtype=np.uint32).ravel()
result = np.zeros(scan + (num_masks,), dtype=np.uint64).ravel()

print("init took %.3fs" % (time.time() - t0))

t1 = time.time()
test.do_apply_mask(images, masks, result,
                   num_masks,
                   detector_shape[0]*detector_shape[1],
                   scan[0]*scan[1])
delta = time.time() - t1
print("delta=%.3f" % delta)

exp_res = detector_shape[0] * detector_shape[1]
for r in result:
    assert r == exp_res, "%r != %r" % (r, exp_res)
