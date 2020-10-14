import os


def warmup_cache(flist):
    for fname in flist:
        with open(fname, "rb") as f:
            while f.read(2**20):
                pass


def drop_cache(flist):
    for fname in flist:
        with open(fname, "rb") as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
