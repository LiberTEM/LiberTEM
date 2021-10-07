import os
import platform

from libertem.io.dataset.base import MMapBackend, BufferedBackend


def warmup_cache(flist):
    for fname in flist:
        with open(fname, "rb") as f:
            while f.read(2**20):
                pass


if platform.system() == "Windows":
    import win32file

    def drop_cache(flist):
        for fname in flist:
            # See https://stackoverflow.com/a/7113153/13082795
            # CreateFile:
            # https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea
            # Direct IO: https://docs.microsoft.com/en-us/windows/win32/fileio/file-buffering
            # FILE_FLAG_NO_BUFFERING opens the file for Direct IO.
            # This drops the cache for this file. This behavior is not explicitly documented,
            # but works consistently and makes sense from a technical point of view:
            # First, writing with Direct IO inherently invalidates the cache, and second,
            # an application that opens a file with Direct IO wants to do its own buffering
            # or doesn't need buffering, so the memory can be freed up for other purposes.
            f = win32file.CreateFile(
                fname,  # fileName
                win32file.GENERIC_READ,  # desiredAccess
                win32file.FILE_SHARE_READ
                | win32file.FILE_SHARE_WRITE
                | win32file.FILE_SHARE_DELETE,  # shareMode
                None,  # attributes
                win32file.OPEN_EXISTING,  # CreationDisposition
                win32file.FILE_FLAG_NO_BUFFERING,  # flagsAndAttributes
                0,  # hTemplateFile
            )
            f.close()
else:
    def drop_cache(flist):
        for fname in flist:
            with open(fname, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def get_testdata_prefixes():
    pathstring = os.environ.get('LT_BENCH_PREFIXES')
    if pathstring is None:
        dirname = os.path.normpath(
            os.path.join(os.path.dirname(__file__), 'data')
        )
        listing = [os.path.join(dirname, p) for p in os.listdir(dirname)]
        prefixes = [p for p in listing if os.path.isdir(p)]
    else:
        prefixes = pathstring.split(';')
        for p in prefixes:
            assert os.path.isdir(p)
    return prefixes


backends_by_name = {
    "mmap": MMapBackend(),
    "mmap_readahead": MMapBackend(enable_readahead_hints=True),
    "buffered": BufferedBackend(),
    "direct": BufferedBackend(direct_io=True),
}
