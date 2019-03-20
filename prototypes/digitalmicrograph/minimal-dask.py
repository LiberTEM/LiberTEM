import multiprocessing
import os
import sys

import dask.distributed as dd

if not hasattr(sys, 'argv'):
    sys.argv  = []

executable = os.path.join(sys.exec_prefix, 'pythonw.exe')

print(executable)

multiprocessing.set_executable(executable)

if __name__ == '__main__':
    cluster = dd.LocalCluster()
    print('local cluster')