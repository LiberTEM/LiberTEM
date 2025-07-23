from typing import Optional
from contextlib import contextmanager


@contextmanager
def use_gpu(device: Optional[int]):
    if device is None:
        # noop
        yield
    else:
        import cupy
        with cupy.cuda.Device(device):
            yield
