import sys
import time
import hyperspy.api as hs

emd_filename_list = sys.argv[1:]
emd_filename_list.sort()

for emd_filename in emd_filename_list:
    t0 = time.time()
    s = hs.load(emd_filename).transpose(signal_axes=(2, 3))
    t1 = time.time()
    result = s.sum()
    t2 = time.time()
    delta = t2 - t0
    print("\n{}".format(emd_filename))
    print("init", t1 - t0)
    print(delta)
    print("{} MB/s (overall)".format(s.data.nbytes / delta / 1024 / 1024))
    print("{} MB/s (without init)".format(s.data.nbytes / (t2 - t1) / 1024 / 1024))
