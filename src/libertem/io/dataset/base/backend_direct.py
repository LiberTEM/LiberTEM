import platform

from .backend import IOBackend
from .backend_buffered import BufferedBackendImpl


class DirectBackend(IOBackend, id_="direct"):
    """
    I/O backend using a direct I/O reading strategy. This currently
    works on Linux and Windows, Mac OS X is not yet supported.

    Use this backend if your data is much larger than your RAM, and
    you have fast enough storage (NVMe RAID, for example).
    In these cases, the :code:`MMapBackend` or :code:`BufferedBackend`
    is not efficient, as the system is constantly under memory pressure.  In
    that case, this backend can perform much better.

    Parameters
    ----------
    max_buffer_size : int
        Maximum buffer size, in bytes. This is passed to the tileshape
        negotiation to select the right depth.
    """
    def __init__(self, max_buffer_size=16*1024*1024):
        self._max_buffer_size = max_buffer_size

    @classmethod
    def from_json(cls, msg):
        """
        Construct an instance from the already-decoded `msg`.
        """
        raise NotImplementedError("TODO! implement me!")

    @classmethod
    def platform_supported(cls):
        return platform.system() in ["Linux", "Windows"]

    def get_impl(self):
        return BufferedBackendImpl(
            max_buffer_size=self._max_buffer_size,
            direct_io=True,
        )
