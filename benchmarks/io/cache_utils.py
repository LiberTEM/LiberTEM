import os
import platform


def warmup_cache(flist):
    for fname in flist:
        with open(fname, "rb") as f:
            while f.read(2**20):
                pass


if platform.system() == "Windows":
    import win32file

    def drop_cache(flist):
        for fname in flist:
            f = win32file.CreateFile(
                fname,  # fileName
                win32file.GENERIC_READ,  # desiredAccess
                win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE | win32file.FILE_SHARE_DELETE,  # shareMode
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
