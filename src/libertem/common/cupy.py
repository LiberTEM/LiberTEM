from typing import Optional
from contextlib import contextmanager


@contextmanager
def use_gpu(device: Optional[int]):
    if device is None:
        # noop
        yield
    else:
        import cupy
        prev_id = cupy.cuda.Device().id
        try:
            cupy.cuda.Device(device).use()
            yield
        finally:
            cupy.cuda.Device(prev_id).use()
