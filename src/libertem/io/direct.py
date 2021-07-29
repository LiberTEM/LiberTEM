import os
import contextlib


@contextlib.contextmanager
def open_direct(path):
    """
    open `path` for reading with O_DIRECT
    """
    fh = os.open(path, os.O_RDONLY | os.O_DIRECT)
    f = open(fh, "rb", buffering=0)
    yield f
    f.close()


def readinto_direct(f, out):
    """
    read from `f` into numpy array `out`, returning a view
    for the read data
    """
    ret = f.readinto(out)
    if ret is None:
        raise OSError("could not readinto()")
    return out[:ret]
