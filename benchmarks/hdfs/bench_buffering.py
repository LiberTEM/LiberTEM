import time
import ctypes
import numpy as np
from contextlib import contextmanager


READ_SIZE = 1024 * 1024
DATASET_SIZE = 2 * 1024 * 1024 * 1024


def get_fs():
    import hdfs3

    class MyHDFile(hdfs3.HDFile):
        def read_into(self, length, out):
            """
            Read ``length`` bytes from the file into the ``out`` buffer

            ``out`` needs to be a ctypes array, for example created
            with ``ctypes.create_string_buffer``, and must be at least ``length`` bytes long.
            """
            _lib = hdfs3.core._lib
            if not _lib.hdfsFileIsOpenForRead(self._handle):
                raise IOError('File not read mode')
            bufsize = length
            bufpos = 0

            while length:
                bufp = ctypes.byref(out, bufpos)
                ret = _lib.hdfsRead(
                    self._fs, self._handle, bufp, ctypes.c_int32(bufsize - bufpos))
                if ret == 0:  # EOF
                    break
                if ret > 0:
                    length -= ret
                    bufpos += ret
                else:
                    raise IOError('Read file %s Failed:' % self.path, -ret)
            return bufpos

    class MyHDFileSystem(hdfs3.HDFileSystem):
        def open(self, path, mode='rb', replication=0, buff=0, block_size=0):
            return MyHDFile(self, path, mode, replication=replication, buff=buff,
                            block_size=block_size)

    return MyHDFileSystem('localhost', port=9000)


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


@contextmanager
def timer(name):
    print("--- starting timer %s ---" % name)
    t1 = time.time()
    yield
    t2 = time.time()
    print("--- stopping timer %s, delta=%0.5f ---" % (name, (t2 - t1)))


def read_old_style(fn):
    fs = get_fs()
    with fs.open(fn) as fd:
        while True:
            data = fd.read(length=READ_SIZE)
            if len(data) == 0:
                break


def read_new_style(fn):
    fs = get_fs()
    with fs.open(fn) as fd:
        buf = ctypes.create_string_buffer(READ_SIZE)
        while True:
            bytes_read = fd.read_into(length=READ_SIZE, out=buf)
            if bytes_read == 0:
                break


def read_new_style_realloc(fn):
    fs = get_fs()
    with fs.open(fn) as fd:
        while True:
            buf = ctypes.create_string_buffer(READ_SIZE)
            bytes_read = fd.read_into(length=READ_SIZE, out=buf)
            if bytes_read == 0:
                break


def read_tests(fn):
    with timer("old style"):
        read_old_style(fn)

    with timer("new style"):
        read_new_style(fn)

    with timer("new style w/ realloc"):
        read_new_style_realloc(fn)


if __name__ == "__main__":
    fn = "hdfs3buffering"
    maybe_create(fn)
    read_tests(fn)
