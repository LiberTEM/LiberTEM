import time
import ctypes
import numpy as np

# make sure that READ_SIZE is <= the hdfs default buffer size, as otherwise
# we can't test the two cases (libhdfs3-internal buffering enabled, disabled)
READ_SIZE = 1 * 1024 * 1024
DATASET_SIZE = 8 * 1024 * 1024 * 1024


def get_fs(config=None):
    import hdfs3

    class MyHDFile(hdfs3.HDFile):
        def old_read(self, length=None):
            """ Read bytes from open file """
            _lib = hdfs3.core._lib
            if not _lib.hdfsFileIsOpenForRead(self._handle):
                raise OSError('File not read mode')
            buffers = []
            buffer_size = self.buff if self.buff != 0 else hdfs3.core.DEFAULT_READ_BUFFER_SIZE

            if length is None:
                out = 1
                while out:
                    out = self.read(buffer_size)
                    buffers.append(out)
            else:
                while length:
                    bufsize = min(buffer_size, length)
                    p = ctypes.create_string_buffer(bufsize)
                    ret = _lib.hdfsRead(
                        self._fs, self._handle, p, ctypes.c_int32(bufsize))
                    if ret == 0:
                        break
                    if ret > 0:
                        if ret < bufsize:
                            buffers.append(p.raw[:ret])
                        elif ret == bufsize:
                            buffers.append(p.raw)
                        length -= ret
                    else:
                        raise OSError('Read file %s Failed:' % self.path, -ret)

            return b''.join(buffers)

    class MyHDFileSystem(hdfs3.HDFileSystem):
        def open(self, path, mode='rb', replication=0, buff=0, block_size=0):
            return MyHDFile(self, path, mode, replication=replication, buff=buff,
                            block_size=block_size)

    defaultconfig = {
        # 'input.localread.default.buffersize': str(1 * 1024 * 1024),
        'input.read.default.verify': '1'
    }

    if config is not None:
        defaultconfig.update(config)

    return MyHDFileSystem('localhost', port=8020, pars=defaultconfig)


def maybe_create(fn):
    fs = get_fs()
    if not fs.exists(fn):
        print("creating test data")
        num = DATASET_SIZE // 8
        data = np.random.rand(num)
        with fs.open(fn, "wb", block_size=data.nbytes) as fd:
            bytes_written = fd.write(data.tobytes())
            assert bytes_written == data.nbytes
        return data


def timer(name, repeats=3):
    def _decorator(fn):
        def _inner(*args, **kwargs):
            deltas = []
            for i in range(repeats):
                t1 = time.time()
                fn(*args, **kwargs)
                t2 = time.time()
                deltas.append(t2 - t1)
            print(f"{name}: {min(deltas):0.5f}")
        return _inner
    return _decorator


@timer("old_read(length=READ_SIZE)")
def read_old_impl(fs, fn):
    with fs.open(fn) as fd:
        while True:
            data = fd.old_read(length=READ_SIZE)
            if len(data) == 0:
                break


@timer("read(length=READ_SIZE)")
def read_new_impl(fs, fn):
    with fs.open(fn) as fd:
        while True:
            data = fd.read(length=READ_SIZE)
            if len(data) == 0:
                break


@timer("read(length=READ_SIZE, out_buffer=buf)")
def read_new_impl_into(fs, fn):
    with fs.open(fn) as fd:
        buf = bytearray(READ_SIZE)
        while True:
            data = fd.read(length=READ_SIZE, out_buffer=buf)
            if data.nbytes == 0:
                break


@timer("read(length=READ_SIZE, out_buffer=True)")
def read_new_impl_true(fs, fn):
    with fs.open(fn) as fd:
        while True:
            data = fd.read(length=READ_SIZE, out_buffer=True)
            if data.nbytes == 0:
                break


def read_tests(fn):
    c1 = {
        'input.read.default.verify': '1'
    }
    c2 = {
        'input.read.default.verify': 'false'
    }
    c3 = {
        'input.localread.default.buffersize': '1',
        'input.read.default.verify': 'false'
    }

    for conf in [c1, c2, c3]:
        print("config: %s" % conf)
        fs = get_fs(conf)
        read_old_impl(fs, fn)
        read_new_impl(fs, fn)
        read_new_impl_true(fs, fn)
        read_new_impl_into(fs, fn)


if __name__ == "__main__":
    fn = "hdfs3buffering"
    maybe_create(fn)
    read_tests(fn)
