import sys
import time
import hyperspy.api as hs
import numpy as np


for filename in sys.argv[1:]:
    t0 = time.time()
    data = np.fromfile(filename).reshape((128, 128, 128, 128))
    s = hs.signals.BaseSignal(data=data).transpose(signal_axes=(2, 3))
    result = hs.signals.BaseSignal(data=np.ndarray((128, 128)))
    t1 = time.time()
    s.sum(out=result)
    t2 = time.time()
    delta = t2 - t0
    print(delta)
    print("init: %.8f" % (t1 - t0))
    print(f"{s.data.nbytes / delta / 1024 / 1024} MB/s (overall)")
    print(f"{s.data.nbytes / (t2 - t1) / 1024 / 1024} MB/s (without init)")
