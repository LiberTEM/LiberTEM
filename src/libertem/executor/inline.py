from typing import Any
from collections.abc import Iterable
import contextlib

import cloudpickle
import psutil
import uuid

from .base import BaseJobExecutor
from libertem.common.executor import (
    Environment, SimpleWorkerQueue, TaskCommHandler, TaskProtocol, WorkerContext, WorkerQueue
)
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.backend import get_use_cuda


class InlineWorkerContext(WorkerContext):
    def __init__(self, queue: SimpleWorkerQueue, msg_queue: SimpleWorkerQueue):
        self._queue = queue
        self._msg_queue = msg_queue

    def get_worker_queue(self) -> WorkerQueue:
        return self._queue

    def signal(self, ident: str, topic: str, msg_dict: dict[str, Any]):
        if 'ident' in msg_dict:
            raise ValueError('ident is a reserved name')
        msg_dict.update({'ident': ident})
        self._msg_queue.put((topic, msg_dict))


class InlineJobExecutor(BaseJobExecutor):
    """
    Naive JobExecutor that just iterates over partitions and processes them one after another

    Parameters
    ----------
    debug : bool
        Set this to enable additional serializability checks

    inline_threads : Optional[int]
        How many fine grained threads should be allowed? Leaving this `None` will
        allow one thread per CPU core
    """
    def __init__(self, debug=False, inline_threads=None, *args, **kwargs):
        # Only import if actually instantiated, i.e. will likely be used
        import libertem.preload  # noqa: 401
        self._debug = debug
        self._inline_threads = inline_threads
        self._scattered = {}

    @contextlib.contextmanager
    def scatter(self, obj):
        obj_id = uuid.uuid4()
        self._scattered[obj_id] = obj
        try:
            yield obj_id
        finally:
            del self._scattered[obj_id]

    def scatter_update(self, handle, obj):
        self._scattered[handle] = obj

    def scatter_update_patch(self, handle, patch):
        self._scattered[handle].patch(patch)

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        task_comm_handler: TaskCommHandler,
    ):
        worker_queue = SimpleWorkerQueue()
        msg_queue = SimpleWorkerQueue()
        task_comm_handler.start()
        threads = self._inline_threads
        if threads is None:
            threads = psutil.cpu_count(logical=False)
        worker_context = InlineWorkerContext(queue=worker_queue,
                                             msg_queue=msg_queue)
        env = Environment(
            threads_per_worker=threads,
            threaded_executor=False,
            worker_context=worker_context,
        )
        with task_comm_handler.monitor(msg_queue):
            for task in tasks:
                if self._debug:
                    cloudpickle.loads(cloudpickle.dumps(task))
                task_comm_handler.handle_task(task, worker_queue)
                result = task(env=env, params=self._scattered[params_handle])
                if self._debug:
                    cloudpickle.loads(cloudpickle.dumps(result))
                yield result, task

        task_comm_handler.done()

    def run_function(self, fn, *args, **kwargs):
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps((fn, args, kwargs)))
        result = fn(*args, **kwargs)
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps(result))
        return result

    def map(self, fn, iterable):
        return [fn(item)
                for item in iterable]

    def run_each_host(self, fn, *args, **kwargs):
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps((fn, args, kwargs)))
        return {"localhost": fn(*args, **kwargs)}

    def run_each_worker(self, fn, *args, **kwargs):
        return {"inline": fn(*args, **kwargs)}

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}
        if get_use_cuda() is not None:
            resources["CUDA"] = 1

        return WorkerSet([
            Worker(name='inline', host='localhost', resources=resources, nthreads=1)
        ])
