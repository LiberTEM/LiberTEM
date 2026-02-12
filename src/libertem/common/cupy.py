from contextlib import contextmanager


# Avoid repeated unsuccessful imports since they
# may try to load sth from file system each time
has_cupy = None


@contextmanager
def use_gpu(device: int | None):
    global has_cupy
    if device is None or has_cupy is False:
        # noop
        yield
    else:
        try:
            # TODO work out how to select devices
            # if cupy is not installed
            import cupy
            prev_id = cupy.cuda.Device().id
            try:
                cupy.cuda.Device(device).use()
                yield
            finally:
                has_cupy = True
                cupy.cuda.Device(prev_id).use()
        except (ImportError, ModuleNotFoundError):
            has_cupy = False
            yield
