import multiprocessing as mp
import contextlib
import math
import queue
from typing import TYPE_CHECKING, Any, NamedTuple, Optional
from collections.abc import Generator

import cloudpickle
import numpy as np

from libertem.common.executor import WorkerQueue, WorkerQueueEmpty

if TYPE_CHECKING:
    from multiprocessing import shared_memory


class PoolAllocation(NamedTuple):
    shm_name: str
    handle: int  # internal handle, could be an offset
    full_size: int  # full size of the allocation, in bytes (including padding)
    req_size: int  # requested allocation size

    def resize(self, new_req_size) -> "PoolAllocation":
        assert new_req_size <= self.full_size
        return PoolAllocation(
            shm_name=self.shm_name,
            handle=self.handle,
            full_size=self.full_size,
            req_size=new_req_size,
        )


class PoolShmClient:
    def __init__(self):
        self._shm_cache = {}

    def get(self, allocation: PoolAllocation) -> memoryview:
        from multiprocessing import shared_memory
        offset = allocation.handle
        name = allocation.shm_name
        if name in self._shm_cache:
            shm = self._shm_cache[name]
        else:
            self._shm_cache[name] = shm = shared_memory.SharedMemory(name=name, create=False)
        return shm.buf[offset:offset+allocation.req_size]


class PoolShmAllocator:
    ALIGN_TO = 4096

    def __init__(self, item_size, size_num_items, create=True, name=None):
        """
        Bump plus free list allocator. Can allocate objects of a fixed size.
        Memory above the `_used` offset is free, allocated blocks from the
        `_free` list are free, everything else is in use.

        Allocation and recycling should happen on the same process.
        """
        self._item_size = item_size
        size = item_size * size_num_items
        size = self.ALIGN_TO * math.ceil(size / self.ALIGN_TO)
        self._create = create
        self._name = name
        self._size = size
        self._free = []  # list of byte offsets into `_shm`
        self._used = 0  # byte offset into `_shm`; everything above is free memory
        self._shm = self._open()

    def _open(self):
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(
            create=self._create,
            name=self._name,
            size=self._size
        )
        return shm

    def allocate(self, req_size: int) -> PoolAllocation:
        if req_size > self._item_size:
            raise RuntimeError(f"allocation request for size {req_size} cannot be serviced")
        # 1) check free list, get an item from there
        if len(self._free) > 0:
            offset = self._free.pop()
        # 2) if free list is empty, bump up `_used` and return a "new" block of
        #    memory
        elif self._used + self._item_size < self._size:
            offset = self._used
            self._used += self._item_size
        else:
            raise RuntimeError("pool shm is out of memory")
        return PoolAllocation(
            shm_name=self._shm.name,
            handle=offset,
            full_size=self._item_size,
            req_size=req_size,
        )

    def get(self, allocation: PoolAllocation) -> memoryview:
        offset = allocation.handle
        assert allocation.shm_name == self._shm.name
        return self._shm.buf[offset:offset+allocation.req_size]

    def recycle(self, allocation: PoolAllocation):
        self._free.append(allocation.handle)

    def shutdown(self):
        self._shm.unlink()

    @property
    def name(self):
        return self._shm.name


def drain_queue(q: mp.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


class ShmQueue(WorkerQueue):
    def __init__(self):
        mp_ctx = mp.get_context("spawn")
        self.q = mp_ctx.Queue()
        self.release_q = mp_ctx.Queue()
        self._pool_shm_allocator = None
        self._pool_shm_client = None
        self._closed = False

    @contextlib.contextmanager
    def put_nocopy(self, header: Any, size: int) -> Generator[memoryview, None, None]:
        alloc_handle, payload_shm = self._get_buf_for_writing(size)
        yield payload_shm
        self.q.put((cloudpickle.dumps(header), 'bytes', alloc_handle))

    def put(self, header, payload: Optional[memoryview] = None):
        """
        Send the (header, payload) tuple via this channel - copying the
        `payload` to a shared memory segment while sending `header` plainly
        via a queue. The header should be `pickle`able.
        """
        if payload is not None:
            payload_shm: Optional[PoolAllocation] = self._copy_to_shm(payload)
        else:
            payload_shm = None
        self.q.put((cloudpickle.dumps(header), 'bytes', payload_shm))

    def _get_buf_for_writing(self, size: int) -> tuple[PoolAllocation, memoryview]:
        if self._pool_shm_allocator is None:
            # FIXME: config item size, pool size
            self._pool_shm_allocator = PoolShmAllocator(
                item_size=512*512*4*2, size_num_items=24*128
            )
        try:
            alloc_handle: PoolAllocation = self.release_q.get_nowait()
            alloc_handle = alloc_handle.resize(size)
        except queue.Empty:
            alloc_handle = self._pool_shm_allocator.allocate(size)
        payload_shm = self._pool_shm_allocator.get(alloc_handle)
        assert payload_shm.nbytes == size, f"{payload_shm.nbytes} != {size}"
        return alloc_handle, payload_shm

    def _copy_to_shm(self, src_buffer: memoryview) -> PoolAllocation:
        """
        Copy the `buffer` to shared memory and return its name
        """
        size = src_buffer.nbytes
        alloc_handle, payload_shm = self._get_buf_for_writing(size)
        src_arr = np.frombuffer(src_buffer, dtype=np.uint8)
        arr_shm = np.frombuffer(payload_shm, dtype=np.uint8)
        assert arr_shm.size == size, f"{arr_shm.size} != {size}"
        arr_shm[:] = src_arr
        return alloc_handle

    def _get_named_shm(self, name: str) -> "shared_memory.SharedMemory":
        from multiprocessing import shared_memory
        return shared_memory.SharedMemory(name=name, create=False)

    @contextlib.contextmanager
    def get(self, block: bool = True, timeout: Optional[float] = None):
        """
        Receive a message. Memory of the payload will be cleaned up after the
        context manager scope, so don't keep references outside of it!

        Parameters
        ----------
        timeout
            Timeout in seconds,
        """
        if self._pool_shm_client is None:
            self._pool_shm_client = PoolShmClient()
        payload_memview: Optional[memoryview] = None
        payload_handle = None
        try:
            header, typ, payload_handle = self.q.get(block=block, timeout=timeout)
            if payload_handle is not None:
                payload_buf = self._pool_shm_client.get(payload_handle)
                payload_memview = memoryview(payload_buf)
            else:
                payload_buf = None
                payload_memview = None
            if typ == "bytes":
                yield (cloudpickle.loads(header), payload_memview)
        except queue.Empty:
            raise WorkerQueueEmpty()
        finally:
            if payload_memview is not None:
                payload_memview.release()
            if payload_handle is not None:
                self.release_q.put(payload_handle)

    def empty(self):
        return self.q.empty()

    def close(self):
        if not self._closed:
            drain_queue(self.q)
            self.q.close()
            self.q.join_thread()
            drain_queue(self.release_q)
            self.release_q.close()
            self.release_q.join_thread()
            self._closed = True
