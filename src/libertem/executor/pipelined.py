import os
import gc
import logging
import functools
import contextlib
import multiprocessing as mp
from typing import (
    Any, Callable, Dict, Iterable, List, NamedTuple, Tuple,
    Generator, TypeVar, TYPE_CHECKING
)
import uuid

import cloudpickle
from opentelemetry import trace
import psutil

from libertem.common.executor import (
    Environment, TaskProtocol, WorkerContext, WorkerQueue,
    WorkerQueueEmpty, MainController,
)
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.tracing import add_partition_to_span, attach_to_parent, maybe_setup_tracing

from .base import BaseJobExecutor

try:
    import prctl
except ImportError:
    prctl = None

if TYPE_CHECKING:
    from .utils.shmqueue import ShmQueue

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


T = TypeVar('T')


class WorkerQueues(NamedTuple):
    request: "ShmQueue"
    response: "ShmQueue"


class WorkerPool:
    """
    Combination of worker processes and matching request queues,
    and a single response queue.

    Processes are started using the spawn method, so they need
    to use primitives that are compatible with spawn.

    Take care to properly close queues and join workers at shutdown.
    """
    def __init__(self, processes: int, worker_fn: Callable):
        from .utils.shmqueue import ShmQueue
        self._workers: List[Tuple[WorkerQueues, mp.Process]] = []
        self._worker_fn = worker_fn
        self._num_processes = processes
        self._response_q = ShmQueue()
        self._mp_ctx = mp.get_context("spawn")
        self._start_workers()

    @property
    def response_queue(self) -> "ShmQueue":
        return self._response_q

    @property
    def size(self) -> int:
        return self._num_processes

    def _start_workers(self):
        with tracer.start_as_current_span("WorkerPool._start_workers"):
            for i in range(self._num_processes):
                queues = self._make_worker_queues()
                p = self._mp_ctx.Process(target=self._worker_fn, args=(queues, i))
                p.start()
                self._workers.append((queues, p))

    def all_worker_queues(self):
        for (qs, _) in self._workers:
            yield qs

    @property
    def workers(self):
        return self._workers

    def join_all(self):
        for (_, p) in self._workers:
            p.join()

    def all_alive(self):
        return all(p.is_alive() for (_, p) in self._workers)

    def close_resp_queue(self):
        self._response_q.close()

    def get_worker_queues(self, idx: int) -> WorkerQueues:
        return self._workers[idx][0]

    def _make_worker_queues(self):
        from .utils.shmqueue import ShmQueue
        return WorkerQueues(
            request=ShmQueue(),
            response=self._response_q,
        )


def set_thread_name(name: str):
    """
    Set a thread name; mostly useful for using system tools for profiling

    Parameters
    ----------
    name : str
        The thread name
    """
    if prctl is None:
        return
    prctl.set_name(name)


def worker_run_task(header, work_mem, queues, worker_idx, env):
    with tracer.start_as_current_span("RUN_TASK") as span:
        try:
            span.set_attributes({
                "libertem.task_size": len(header["task"]),
            })
            gc.disable()
            task: TaskProtocol = cloudpickle.loads(header["task"])
            params_handle = header["params_handle"]
            params = work_mem[params_handle]
            partition = task.get_partition()
            add_partition_to_span(partition)
            result = task(params, env)
            queues.response.put({
                "type": "RESULT",
                "result": result,
                "task_id": header["task_id"],
                "worker_id": worker_idx,
            })
        except Exception as e:
            logger.exception("failure in RUN_TASK")
            queues.response.put({
                "type": "ERROR",
                "error": e,
                "worker_id": worker_idx,
            })
        finally:
            gc.enable()
            with tracer.start_as_current_span("gc"):
                gc.collect()


def worker_run_function(header, queues, idx):
    with tracer.start_as_current_span("RUN_FUNCTION"):
        fn = cloudpickle.loads(header["fn"])
        try:
            result = fn()
            queues.response.put({
                "type": "RUN_FUNCTION_RESULT",
                "result": result,
                "worker_id": idx,
            })
        except Exception as e:
            logger.exception("failure in RUN_FUNCTION")
            queues.response.put({
                "type": "ERROR",
                "error": e,
                "worker_id": idx,
            })


def worker_loop(queues, work_mem, worker_idx, env):
    while True:
        with queues.request.get() as msg:
            header, payload = msg
            header_type = header["type"]
            if header_type == "RUN_TASK":
                with attach_to_parent(header["span_context"]):
                    worker_run_task(header, work_mem, queues, worker_idx, env)
            elif header_type == "SCATTER":
                # FIXME: array data could be transferred and stored in SHM instead
                key = header["key"]
                assert key not in work_mem
                work_mem[key] = header["value"]
                continue
            elif header_type == "RUN_FUNCTION":
                with attach_to_parent(header["span_context"]):
                    worker_run_function(header, queues, worker_idx)
            elif header_type == "CLEANUP":
                key = header["key"]
                assert key in work_mem
                del work_mem[key]
                continue
            elif header_type == "SHUTDOWN":
                queues.request.close()
                queues.response.close()
                break
            elif header_type == "WARMUP":
                with attach_to_parent(header["span_context"]):
                    with tracer.start_as_current_span("WARMUP"):
                        with env.enter():
                            pass
            else:
                queues.response.put({
                    "type": "ERROR",
                    "error": f"unknown message {header}",
                    "worker_id": worker_idx,
                })
                continue


def pipelined_worker(queues: WorkerQueues, worker_idx: int, pin: bool):
    try:
        if hasattr(os, 'sched_setaffinity') and pin:
            os.sched_setaffinity(0, [worker_idx])
        work_mem: Dict[str, Any] = {}

        set_thread_name(f"worker-{worker_idx}")

        # FIXME: propagate exporter settings to the workers, too!
        maybe_setup_tracing(service_name="pipelined_worker", service_id=f"worker-{worker_idx}")

        worker_context = PipelinedWorkerContext(queues.request)
        env = Environment(
            threaded_executor=False,
            threads_per_worker=1,
            worker_context=worker_context
        )

        queues.response.put({
            "type": "STARTUP_DONE",
            "worker_id": worker_idx,
        })

        return worker_loop(queues, work_mem, worker_idx, env)
    except Exception as e:
        queues.response.put({
            "type": "ERROR",
            "worker_id": worker_idx,
            "error": e,
        })
        # FIXME: do we need to explicitly join the queue(s) here, too?


class PipelinedWorkerContext(WorkerContext):
    def __init__(self, queue: "ShmQueue"):
        self._queue = queue

    def get_worker_queue(self) -> WorkerQueue:
        return self._queue


ResultT = Generator[Tuple[Any, TaskProtocol], None, None]
ResultWithID = Generator[Tuple[Any, TaskProtocol, int], None, None]


def _order_results(results_in: ResultWithID) -> ResultT:
    last_sent_id = -1
    # for results that are received out-of-order, keep sorted:
    result_stack: List[Tuple[Any, TaskProtocol, int]] = []
    span = trace.get_current_span()
    for result, task, task_id in results_in:
        if task_id == last_sent_id + 1:
            span.add_event("_order_results.yield")
            yield (result, task)
            last_sent_id = task_id
        else:
            span.add_event("_order_results.postpone", {
                "expect": last_sent_id + 1,
                "is": task_id,
            })
            result_stack = sorted(
                result_stack + [(result, task, task_id)],
                key=lambda x: x[-1]
            )
        # top of the stack looks good, yield as long as it matches:
        while len(result_stack) > 0 and result_stack[0][2] == last_sent_id + 1:
            res = result_stack.pop(0)
            yield res[0], res[1]
            last_sent_id = res[2]

    for result, task, task_id in result_stack:
        if task_id != last_sent_id + 1:
            raise RuntimeError(
                f"missing tasks? end of result but next id on result_stack is "
                f"{task_id}, was expecting {last_sent_id + 1}"
            )
        span.add_event("_order_results.yield")
        yield (result, task)
        last_sent_id = task_id


class PipelinedExecutor(BaseJobExecutor):
    def __init__(
        self,
        n_workers: int = None,
        pin_workers: bool = True,
    ) -> None:
        if n_workers is None:
            n_workers = psutil.cpu_count(logical=False)
        self._n_workers = n_workers
        self._pin_workers = pin_workers
        self._pool = self.start_pool()

    def start_pool(self) -> WorkerPool:
        with tracer.start_as_current_span("PipelinedExecutor.start_pool") as span:
            pool = WorkerPool(
                processes=self._n_workers,
                worker_fn=functools.partial(
                    pipelined_worker, pin=self._pin_workers,
                )
            )
            for i in range(pool.size):
                with pool.response_queue.get() as (msg, _):
                    if msg["type"] == "ERROR":
                        raise RuntimeError(
                            f"error on startup: {msg['error']}"
                        )
                    if msg["type"] != "STARTUP_DONE":
                        raise RuntimeError(
                            f"unknown message type {msg['type']}, expected STARTUP_DONE"
                        )
                    span.add_event("worker startup done", {"worker_id": msg["worker_id"]})
            return pool

    def _run_tasks_inner(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        controller: "MainController",
    ) -> ResultWithID:
        in_flight = 0
        id_to_task = {}

        controller.start()
        span = trace.get_current_span()
        span_context = span.get_span_context()

        for qs, _ in self._pool.workers:
            qs.request.put({
                "type": "WARMUP",
                "span_context": span_context,
            })

        for task_idx, task in enumerate(tasks):
            in_flight += 1
            id_to_task[task_idx] = task

            worker_idx = task_idx % self._pool.size
            worker_queues = self._pool.get_worker_queues(worker_idx)
            worker_queues.request.put({
                "type": "RUN_TASK",
                "task": cloudpickle.dumps(task),
                "task_id": task_idx,
                "params_handle": params_handle,
                "span_context": span_context,
            })
            # FIXME: semantics of this - is this enough?
            # does it matter if this is enough? we can change it in the future if not
            # could be: the function returns once it has forwarded
            # all the data necessary for the given task,
            # (or, in the offline case, immediately)
            controller.handle_task(task, worker_queues.request)

            try:
                with self._pool.response_queue.get(block=False) as (result, _):
                    if result["type"] == "ERROR":
                        raise RuntimeError(f"failed to run tasks: {result['error']}")
                    in_flight -= 1
                    result_task_id = result["task_id"]
                    yield (result["result"], id_to_task[result_task_id], result_task_id)
            except WorkerQueueEmpty:
                continue

        # at the end, block to get the remaining results:
        while in_flight > 0:
            try:
                with self._pool.response_queue.get() as (result, _):
                    if result["type"] == "ERROR":
                        raise RuntimeError(f"failed to run tasks: {result['error']}")
                    in_flight -= 1
                    result_task_id = result["task_id"]
                    # FIXME: adjust order of results to match tasks
                    # idea: if in-order, just yield result, otherwise, put
                    # result into an ordered queue, and yield out "contiguous"
                    # runs of results as soon as they are available.
                    yield (result["result"], id_to_task[result_task_id], result_task_id)
            except WorkerQueueEmpty:
                continue

        controller.done()

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        controller: "MainController",
    ) -> ResultT:
        with tracer.start_as_current_span("PipelinedExecutor.run_tasks"):
            yield from _order_results(self._run_tasks_inner(
                tasks, params_handle, cancel_id, controller,
            ))

    def get_available_workers(self) -> WorkerSet:
        resources = {"compute": 1, "CPU": 1}
        return WorkerSet([
            Worker(
                name="cpu-%d" % idx,
                host="localhost",
                resources=resources,
                nthreads=1,
            )
            for idx, worker in enumerate(self._pool.workers)
        ])

    def close(self):
        self._pool.close_resp_queue()
        for qs, p in self._pool.workers:
            qs.request.put({"type": "SHUTDOWN"})
            qs.request.close()
            qs.response.close()
            p.join()

    def _run_function(self, fn: Callable[..., T], worker_idx, *args, **kwargs) -> T:
        qs = self._pool.get_worker_queues(worker_idx)
        f = functools.partial(fn, *args, **kwargs)
        pickled = cloudpickle.dumps(f)
        qs.request.put({
            "type": "RUN_FUNCTION",
            "fn": pickled,
            "span_context": trace.get_current_span().get_span_context(),
        })
        # FIXME: timeout?
        with qs.response.get() as (response, _):
            if not response["type"] == "RUN_FUNCTION_RESULT":
                raise RuntimeError(f"invalid response type: {response['TYPE']}")
            result: T = response["result"]
            return result

    def run_function(self, fn: Callable[..., T], *args, **kwargs) -> T:
        # FIXME: this is not concurrency-safe currently! beware!
        with tracer.start_as_current_span("PipelinedExecutor.run_tasks"):
            return self._run_function(fn, 0, *args, **kwargs)

    def run_each_worker(self, fn, *args, **kwargs):
        # FIXME: not as fast as it could be, but also not really perf-sensitive?
        result = {}
        for idx, worker in enumerate(self.get_available_workers()):
            result[worker.name] = self._run_function(fn, worker_idx=idx, *args, **kwargs)
        return result

    def run_each_host(self, fn, *args, **kwargs):
        return {"localhost": self.run_function(fn, *args, **kwargs)}

    @contextlib.contextmanager
    def scatter(self, obj):
        key = str(uuid.uuid4())
        for qs, p in self._pool.workers:
            qs.request.put({
                "type": "SCATTER",
                "key": key,
                "value": obj,
            })
        yield key
        for qs, p in self._pool.workers:
            qs.request.put({
                "type": "CLEANUP",
                "key": key,
            })

    def map(self, fn, iterable):
        # FIXME: replace with efficient impl if needed
        with tracer.start_as_current_span("PipelinedExecutor.map"):
            return [
                self._run_function(fn=lambda: fn(item), worker_idx=0)
                for item in iterable
            ]
