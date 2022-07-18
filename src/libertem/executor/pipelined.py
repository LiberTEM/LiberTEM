import os
import gc
import sys
import logging
import functools
import contextlib
import multiprocessing as mp
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Iterable, List, NamedTuple, Tuple,
    Generator, TypeVar, Union,
)
from typing_extensions import TypedDict, Literal
import uuid

import cloudpickle
from opentelemetry import trace
from libertem.common.backend import set_use_cpu, set_use_cuda

from libertem.common.executor import (
    Environment, TaskProtocol, WorkerContext, WorkerQueue,
    WorkerQueueEmpty, MainController, SimpleMPWorkerQueue,
)
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.tracing import add_partition_to_span, attach_to_parent, maybe_setup_tracing

from .base import BaseJobExecutor

try:
    import prctl
except ImportError:
    prctl = None

if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


T = TypeVar('T')


class WorkerQueues(NamedTuple):
    request: "WorkerQueue"
    response: "WorkerQueue"


class WorkerSpec(TypedDict):
    name: str
    device_id: int
    device_kind: Union[Literal['CPU'], Literal['CUDA']]
    worker_idx: int


class WorkerPool:
    """
    Combination of worker processes and matching request queues,
    and a single response queue.

    Processes are started using the spawn method, so they need
    to use primitives that are compatible with spawn.

    Take care to properly close queues and join workers at shutdown.
    """
    def __init__(self, worker_fn: Callable, spec: List[WorkerSpec]):
        self._worker_q_cls = SimpleMPWorkerQueue
        self._workers: List[Tuple[WorkerQueues, mp.Process]] = []
        self._worker_fn = worker_fn
        self._response_q = self._worker_q_cls()
        self._mp_ctx = mp.get_context("spawn")
        self._spec = spec
        self._start_workers()

    @property
    def response_queue(self) -> "WorkerQueue":
        return self._response_q

    @property
    def size(self) -> int:
        return len(self._spec)

    def _start_workers(self):
        with tracer.start_as_current_span("WorkerPool._start_workers") as span:
            span_context = span.get_span_context()
            for spec_item in self._spec:
                queues = self._make_worker_queues()
                p = self._mp_ctx.Process(target=self._worker_fn, kwargs={
                    "queues": queues,
                    "spec": spec_item,
                    "span_context": span_context,
                })
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
        while True:
            try:
                with self._response_q.get(block=False):
                    continue
            except WorkerQueueEmpty:
                break
        self._response_q.close()

    def get_worker_queues(self, idx: int) -> WorkerQueues:
        return self._workers[idx][0]

    def _make_worker_queues(self):
        return WorkerQueues(
            request=self._worker_q_cls(),
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
                "libertem.task_size_pickled": len(header["task"]),
                "libertem.os.pid": os.getpid(),
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
                    # NOTE: in case of an error, need to drain the request queue
                    # (anything that is left over from the detector-specific
                    # data that was sent in `MainController.handle_task`):
                    with tracer.start_as_current_span("drain after task") as span:
                        while True:
                            with queues.request.get() as msg:
                                header, payload = msg
                                header_type = header["type"]
                                span.add_event("msg", {"type": header_type})
                                assert header_type not in (
                                    "RUN_TASK", "SCATTER", "RUN_FUNCTION",
                                    "CLEANUP", "SHUTDOWN", "WARMUP",
                                )
                                if header_type == "END_TASK":
                                    break
            elif header_type == "SCATTER":
                # FIXME: array data could be transferred and stored in SHM instead
                key = header["key"]
                assert key not in work_mem
                value = header["value"]
                work_mem[key] = value
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
                with attach_to_parent(header["span_context"]):
                    with tracer.start_as_current_span("SHUTDOWN") as span:
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


DeviceT = Tuple[int, Union[Literal['CUDA'], Literal['CPU']]]


def _setup_device(spec: WorkerSpec, pin: bool):
    if spec["device_kind"].lower() == "cpu":
        if hasattr(os, 'sched_setaffinity') and pin:
            os.sched_setaffinity(0, [spec["device_id"]])
        set_use_cpu(spec["device_id"])
    elif spec["device_kind"].lower() == "cuda":
        set_use_cuda(spec["device_id"])


def _drain_and_close(queues: WorkerQueues):
    while True:
        try:
            with queues.request.get(block=False):
                continue
        except WorkerQueueEmpty:
            break
    # close and join:
    queues.request.close()
    queues.response.close()


def pipelined_worker(
    queues: WorkerQueues,
    pin: bool,
    spec: WorkerSpec,
    span_context: "SpanContext",
):
    # FIXME: propagate to parent process with a pipe or similar?
    sys.stderr.close()
    sys.stderr = open(os.open(os.devnull, os.O_RDWR), closefd=False)

    try:
        # FIXME: explicitly propagate exporter settings to the workers?
        # right now taken care of by environment variables...
        worker_idx = spec["worker_idx"]
        maybe_setup_tracing(service_name="pipelined_worker", service_id=f"worker-{worker_idx}")

        # attach to parent span context for startup:
        with attach_to_parent(span_context),\
                tracer.start_as_current_span("pipelined_worker.startup") as span:
            _setup_device(spec, pin)
            work_mem: Dict[str, Any] = {}

            span.set_attributes({
                "libertem.spec.name": spec["name"],
                "libertem.spec.device_id": spec["device_id"],
                "libertem.spec.devide_kind": spec["device_kind"],
                "libertem.spec.worker_idx": spec["worker_idx"],
            })

            set_thread_name(f"worker-{worker_idx}")

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
        _drain_and_close(queues)


class PipelinedWorkerContext(WorkerContext):
    def __init__(self, queue: "WorkerQueue"):
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


def make_spec(
    cpus: Iterable[int],
    cudas: Iterable[int],
    workers_per_gpu: int = 2,
    has_cupy: bool = False,  # currently ignored, for convenience of passing **detect()
) -> List[WorkerSpec]:
    """
    Takes the output of `libertem.utils.devices.detect`
    and makes a plan for starting workers on them.

    Additionally can start multiple workers per GPU
    """
    spec = []
    worker_idx = 0
    for device_id in cpus:
        spec.append(WorkerSpec(
            name=f"cpu-{device_id}",
            device_id=device_id,
            device_kind="CPU",
            worker_idx=worker_idx,
        ))
        worker_idx += 1
    for device_id in cudas:
        for i in range(workers_per_gpu):
            spec.append(WorkerSpec(
                name=f"cuda-{device_id}-{i}",
                device_id=device_id,
                device_kind="CUDA",
                worker_idx=worker_idx,
            ))
            worker_idx += 1
    return spec


class PipelinedExecutor(BaseJobExecutor):
    def __init__(
        self,
        spec: List[WorkerSpec] = None,
        pin_workers: bool = True,
    ) -> None:
        self._pin_workers = pin_workers
        if spec is None:
            spec = self._default_spec()
        self._spec = spec
        self._pool = self.start_pool()
        self._closed = False

        # FIXME: should this be configurable?
        # timeout for cleanup, either from exception or when joining processes
        self._cleanup_timeout = 10.0

    def start_pool(self) -> WorkerPool:
        with tracer.start_as_current_span("PipelinedExecutor.start_pool") as span:
            pool = WorkerPool(
                worker_fn=functools.partial(
                    pipelined_worker, pin=self._pin_workers,
                ),
                spec=self._spec,
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

    @classmethod
    def _default_spec(cls):
        from libertem.utils.devices import detect
        detected = detect()
        return make_spec(**detected)

    @classmethod
    def make_local(cls, **kwargs):
        spec = cls._default_spec()
        return cls(spec=spec, **kwargs)

    @classmethod
    def make_spec(cls, *args, **kwargs):
        return make_spec(*args, **kwargs)

    def _run_tasks_inner(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        controller: "MainController",
    ) -> ResultWithID:
        in_flight = 0
        id_to_task = {}

        try:
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

                # NOTE: sentinel message; in case of errors, the worker
                # needs to discard the data from the queue until it receives
                # this message:
                worker_queues.request.put({
                    "type": "END_TASK",
                    "task_id": task_idx,
                    "params_handle": params_handle,
                    "span_context": span_context,
                })

                try:
                    with self._pool.response_queue.get(block=False) as (result, _):
                        in_flight -= 1
                        if result["type"] == "ERROR":
                            raise RuntimeError(f"failed to run tasks: {result['error']}")
                        result_task_id = result["task_id"]
                        yield (result["result"], id_to_task[result_task_id], result_task_id)
                except WorkerQueueEmpty:
                    continue

            # FIXME: code duplication
            # at the end, block to get the remaining results:
            while in_flight > 0:
                try:
                    with self._pool.response_queue.get() as (result, _):
                        in_flight -= 1
                        if result["type"] == "ERROR":
                            raise RuntimeError(f"failed to run tasks: {result['error']}")
                        result_task_id = result["task_id"]
                        yield (result["result"], id_to_task[result_task_id], result_task_id)
                except WorkerQueueEmpty:
                    continue

            controller.done()
        except Exception:
            # In case of an exception, we need to drain the response queue,
            # so the next `run_tasks` call isn't polluted by old responses.
            # -> just like in the happy case, we need to wait for responses
            # for all in-flight tasks. In case the error happened between incrementing
            # `in_flight` and actually sending the task to the queue, we should
            # have a timeout here to not wait infinitely long.
            while in_flight > 0:
                try:
                    with self._pool.response_queue.get(
                        timeout=self._cleanup_timeout
                    ) as (result, _):
                        in_flight -= 1
                        # we only raise the first exception; log the others here:
                        if result["type"] == "ERROR":
                            logger.error(f"Error response from worker: {result['error']}")
                except WorkerQueueEmpty:
                    continue
            # if from a worker, this is the first exception that got put into the queue
            raise

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
        with tracer.start_as_current_span("PipelinedExecutor.close") as span:
            for idx, (qs, p) in enumerate(self._pool.workers):
                span.add_event("sending SHUTDOWN", {"idx": idx})
                qs.request.put({
                    "type": "SHUTDOWN",
                    "span_context": span.get_span_context(),
                })
                span.add_event("SHUTDOWN sent", {"idx": idx})
                while True:
                    try:
                        with qs.response.get(block=False) as msg:
                            logger.warning(f"got message on close: {msg[0]}")
                    except WorkerQueueEmpty:
                        break
                # FIXME: forcefully kill the process
                # (might need to first send SHUTDOWN to all workers, as
                # killing a process might make the request queue unusable, too)
                # -> basically, need to kill all workers in this case
                # for now, let the timeout bubble up
                p.join(timeout=self._cleanup_timeout)
                qs.request.close()
            self._pool.close_resp_queue()
            self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()

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
