import itertools
import os
import sys
import logging
import functools
import contextlib
import multiprocessing as mp
from typing import (
    TYPE_CHECKING, Any, Callable, NamedTuple, Optional, TypeVar, Union,
)
from collections.abc import Iterable, Generator
from typing_extensions import TypedDict, Literal
import uuid
import warnings
import time

from tblib import pickling_support
import cloudpickle
from opentelemetry import trace
from libertem.common.backend import set_use_cpu, set_use_cuda, set_file_limit

from libertem.common.executor import (
    Environment, TaskProtocol, WorkerContext, WorkerQueue,
    WorkerQueueEmpty, TaskCommHandler, SimpleMPWorkerQueue,
    JobCancelledError, ResourceDef,
)
from libertem.common.scheduler import Worker, WorkerSet, Scheduler
from libertem.common.tracing import add_partition_to_span, attach_to_parent, maybe_setup_tracing

from .utils import assign_cudas
from .base import BaseJobExecutor, ResourceError

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
    message: "WorkerQueue"


class WorkerSpec(TypedDict):
    name: str
    device_id: int
    device_kind: Union[Literal['CPU'], Literal['CUDA']]
    worker_idx: int
    has_cupy: bool


class PoolWorkerInfo(NamedTuple):
    queues: WorkerQueues
    process: mp.Process
    spec: WorkerSpec


class PoolStateError(Exception):
    """
    The worker pool is not in the expected state to perform the requested operation
    """
    pass


def _resources_for_spec(spec: WorkerSpec) -> ResourceDef:
    worker_resources: "ResourceDef" = {}

    if spec['device_kind'] == 'CPU':
        worker_resources["compute"] = 1
        worker_resources["CPU"] = 1
        worker_resources["ndarray"] = 1

    if spec["device_kind"] == "CUDA":
        worker_resources["compute"] = 1
        worker_resources["CUDA"] = 1
        if spec["has_cupy"]:
            worker_resources["ndarray"] = 1

    return worker_resources


class WorkerPool:
    """
    Combination of worker processes and matching request queues,
    and a single response queue.

    Processes are started using the spawn method, so they need
    to use primitives that are compatible with spawn.

    Take care to properly close queues and join workers at shutdown.

    Note
    ----
    We are not using the vanilla :class:`multiprocesing.Pool` here, because
    we need to coordinate the sending and receiving side for each task,
    and we need to keep state on the worker, pin processes to cores etc.
    """
    def __init__(self, worker_fn: Callable, spec: list[WorkerSpec]):
        self._worker_q_cls = SimpleMPWorkerQueue
        self._workers: Optional[list[PoolWorkerInfo]] = []
        self._worker_fn = worker_fn
        self._response_q: Optional[SimpleMPWorkerQueue] = self._worker_q_cls()
        self._message_q = self._worker_q_cls()
        self._mp_ctx = mp.get_context("spawn")
        self._spec = spec
        self._start_workers()

    @property
    def response_queue(self) -> "WorkerQueue":
        if self._response_q is None:
            raise PoolStateError("Response queue not available")
        return self._response_q

    @property
    def message_queue(self) -> "WorkerQueue":
        return self._message_q

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
                self._workers.append(
                    PoolWorkerInfo(
                        queues=queues,
                        process=p,
                        spec=spec_item,
                    )
                )

    def kill_worker(self, worker_info: PoolWorkerInfo, timeout: float = 5.0):
        worker_info.queues.request.close(drain=False, force=True)
        worker_info.process.terminate()
        worker_info.process.join(timeout)
        if worker_info.process.exitcode is None:
            worker_info.process.kill()
            # reap the dead process:
            worker_info.process.join(30)

    def kill(self, timeout: float = 5):
        if self._workers is None:
            raise PoolStateError("Cannot kill workers, not running")
        for worker in self._workers:
            self.kill_worker(worker, timeout=timeout)
        exitcodes = [
            worker.process.exitcode
            for worker in self._workers
        ]
        self._workers = None
        if self._response_q is None:
            raise PoolStateError("Response queue not available")
        self._response_q.close(drain=False, force=True)
        self._response_q = None
        if any([e is None for e in exitcodes]):
            raise PoolStateError("At least one worker failed to be killed")

    @property
    def workers(self) -> list[PoolWorkerInfo]:
        if self._workers is None:
            raise PoolStateError("No workers are running")
        return self._workers

    def all_alive(self) -> bool:
        if self._workers is None:
            raise PoolStateError("No workers are running")
        return all(qp.process.is_alive() for qp in self._workers)

    def assert_all_alive(self):
        """
        Check if any workers are dead, if so, kill the whole pool and raise a
        `RuntimeError`.
        """
        if not self.all_alive():
            self.kill()
            raise RuntimeError(
                "One or more workers failed to start"
            )

    def close_resp_queue(self):
        self._response_q.close()

    def close_mesg_queue(self):
        self._message_q.close()

    def get_worker_queues(self, worker_idx: int) -> WorkerQueues:
        if self._workers is None:
            raise PoolStateError("No workers are running")
        return self._workers[worker_idx].queues

    def _make_worker_queues(self):
        return WorkerQueues(
            request=self._worker_q_cls(),
            response=self._response_q,
            message=self._message_q,
        )


WorkerSelector = Callable[
    [list[PoolWorkerInfo], WorkerPool, TaskProtocol, int],
    PoolWorkerInfo
]


def _select_by_queue_size(
    eligible: list[PoolWorkerInfo],
    pool: WorkerPool,
    task: TaskProtocol,
    task_idx: int,
) -> PoolWorkerInfo:
    # FIXME: I think this introduces some latency currently,
    # as the workers `get` from their queue as soon as possible,
    # so the following could happen:
    # - task 0 becomes available
    # - all queues have size 0, so task gets assigned to worker 0
    # - worker 0 picks the task from its queue, so its queue size is 0 again
    # - task 1 becomes available
    # - all queues have size 0 (again), so task gets assigned to worker 0
    # => this should only happen in the offline case, as in the live
    #    processing case, we push more than one message per task into the
    #    queue, so there should be more than one message in the backlog
    #    when the `TaskCommHandler` is finished pushing messages for one task.
    return min(eligible, key=lambda w: w.queues.request.size())


def _select_by_round_robin(
    eligible: list[PoolWorkerInfo],
    pool: WorkerPool,
    task: TaskProtocol,
    task_idx: int,
) -> PoolWorkerInfo:
    return eligible[task_idx % len(eligible)]


def schedule_task(
    task_idx: int,
    task: TaskProtocol,
    pool: WorkerPool,
    scheduler: Scheduler,
    pool_info_for_worker: dict[Worker, PoolWorkerInfo],
    selector: WorkerSelector,
) -> tuple[int, WorkerQueues]:
    """
    Returns the worker index and its queues of the worker the given task should
    be scheduled on.

    Workers are first filtered based on requested resources, then the concrete
    worker is selected by the function specified in the :code:`selector`
    parameter.
    """

    eligible = [
        pool_info_for_worker[w]
        for w in scheduler.workers_for_task(task)
    ]
    if len(eligible) == 0:
        resources = task.get_resources()
        raise ResourceError(f"No worker available with resources {resources}")

    worker = selector(eligible, pool, task, task_idx)

    idx = pool.workers.index(worker)
    return idx, worker.queues


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


def worker_run_task(
    header: dict,
    work_mem: dict,
    queues: WorkerQueues,
    worker_idx: int,
    env: Environment,
):
    """
    Called from the worker main loop when a RUN_TASK message is received

    Parameters
    ----------

    header
        The header of the message that was received

    work_mem
        The worker's working memory, for accessing scattered data

    queues
        request and response queues

    worker_idx
        This worker's index

    env
        The Environment for preparing thread counts etc. for the UDF run
    """
    with tracer.start_as_current_span("RUN_TASK") as span:
        try:
            span.set_attributes({
                "libertem.task_size_pickled": len(header["task"]),
                "libertem.os.pid": os.getpid(),
            })
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
                "uuid": header["uuid"],
                "worker_id": worker_idx,
            })
        except Exception as e:
            logger.exception("failure in RUN_TASK")
            pickling_support.install(e)
            queues.response.put({
                "type": "ERROR",
                "error": e,
                "exception": e,
                "worker_id": worker_idx,
                "uuid": header["uuid"],
            })
        finally:
            pass


def worker_run_function(header, queues, worker_idx):
    """
    Called from the worker main loop when a RUN_FUNCTION message is received

    Parameters
    ----------
    header
        The header of the message that was received

    queues
        request and response queues

    worker_idx
        This worker's index
    """
    with tracer.start_as_current_span("RUN_FUNCTION"):
        try:
            fn = cloudpickle.loads(header["fn"])
            result = fn()
            queues.response.put({
                "type": "RUN_FUNCTION_RESULT",
                "result": result,
                "worker_id": worker_idx,
            })
        except Exception as e:
            logger.exception("failure in RUN_FUNCTION")
            pickling_support.install(e)
            queues.response.put({
                "type": "ERROR",
                "error": e,
                "exception": e,
                "worker_id": worker_idx,
            })


def _drain_after_task(queues: WorkerQueues):
    """
    In case of an error, need to drain the request queue
    (anything that is left over from the detector-specific
    data that was sent in `TaskCommHandler.handle_task`):
    """
    with tracer.start_as_current_span("drain after task") as span:
        while True:
            with queues.request.get() as msg:
                header, payload = msg
                header_type = header["type"]
                span.add_event("msg", {"type": header_type})
                if header_type in (
                    "RUN_TASK", "SCATTER", "SCATTER_UPDATE",
                    "SCATTER_UPDATE_PATCH", "RUN_FUNCTION",
                    "DELETE", "SHUTDOWN", "WARMUP",
                ):
                    raise RuntimeError(
                        f"unexpected message type {header_type}"
                    )
                if header_type == "END_TASK":
                    break


def worker_loop(
    queues: WorkerQueues,
    work_mem: dict,
    worker_idx: int,
    env: Environment
):
    """
    The worker main loop, called when the worker setup is done.
    Waits for messages on the request queue until a SHUTDOWN message
    is received.

    Parameters
    ----------

    work_mem
        The worker's working memory, for accessing scattered data

    queues
        request and response queues

    worker_idx
        This worker's index

    env
        The Environment for preparing thread counts etc. for the UDF run
    """
    while True:
        try:
            with queues.request.get() as msg:
                header, payload = msg
                header_type = header["type"]
                if header_type == "RUN_TASK":
                    with attach_to_parent(header["span_context"]):
                        worker_run_task(header, work_mem, queues, worker_idx, env)
                        _drain_after_task(queues)
                elif header_type == "SCATTER":
                    # FIXME: array data could be transferred and stored in SHM instead
                    key = header["key"]
                    if key in work_mem:
                        queues.response.put({
                            "type": "ERROR",
                            "error": f"key {key} already stored in worker memory",
                            "worker_id": worker_idx,
                        })
                        continue
                    work_mem[key] = header["value"]
                    continue
                elif header_type == "SCATTER_UPDATE":
                    # FIXME: array data could be transferred and stored in SHM instead
                    key = header["key"]
                    if key not in work_mem:
                        queues.response.put({
                            "type": "ERROR",
                            "error": f"key {key} not stored in worker memory, can't update",
                            "worker_id": worker_idx,
                        })
                        continue
                    work_mem[key] = header["value"]
                    continue
                elif header_type == "SCATTER_UPDATE_PATCH":
                    # FIXME: array data could be transferred and stored in SHM instead
                    key = header["key"]
                    if key not in work_mem:
                        queues.response.put({
                            "type": "ERROR",
                            "error": f"key {key} not stored in worker memory, can't update",
                            "worker_id": worker_idx,
                        })
                        continue
                    if not hasattr(work_mem[key], 'patch'):
                        queues.response.put({
                            "type": "ERROR",
                            "error": f"key {key} is not patcheable",
                            "worker_id": worker_idx,
                        })
                        continue
                    work_mem[key].patch(header["patch"])
                    continue
                elif header_type == "RUN_FUNCTION":
                    with attach_to_parent(header["span_context"]):
                        worker_run_function(header, queues, worker_idx)
                elif header_type == "DELETE":
                    key = header["key"]
                    if key in work_mem:
                        del work_mem[key]
                    continue
                elif header_type == "SHUTDOWN":
                    with attach_to_parent(header["span_context"]):
                        with tracer.start_as_current_span("SHUTDOWN"):
                            queues.request.close(force=True)
                            queues.response.close(drain=False, force=True)
                            queues.message.close(drain=False, force=True)
                        break
                elif header_type == "WARMUP":
                    with attach_to_parent(header["span_context"]):
                        with tracer.start_as_current_span("WARMUP"):
                            import libertem.udf.base  # NOQA
                            import libertem.api  # NOQA
                            import libertem.preload  # NOQA
                            with env.enter():
                                pass
                else:
                    queues.response.put({
                        "type": "ERROR",
                        "error": f"unknown message {header}",
                        "worker_id": worker_idx,
                    })
                    # probably desynchronized with the main process, so give up:
                    raise RuntimeError(f"unknown message, shutting down worker {worker_idx}")
        except KeyboardInterrupt:
            queues.response.put({
                "type": "ERROR",
                "error": "KeyboardInterrupt",
                "worker_id": worker_idx,
            })
            raise


DeviceT = tuple[int, Union[Literal['CUDA'], Literal['CPU']]]


def _setup_device(spec: WorkerSpec, pin: bool):
    """
    Set up this worker for its given task - either CPU or GPU comptation,
    and maybe pin CPU workers to a given CPU core.
    """
    if spec["device_kind"].lower() == "cpu":
        if hasattr(os, 'sched_setaffinity') and pin:
            os.sched_setaffinity(0, [spec["device_id"]])
        set_use_cpu(spec["device_id"])
    elif spec["device_kind"].lower() == "cuda":
        set_use_cuda(spec["device_id"])
    else:
        raise RuntimeError(f"unknown device kind: {spec['device_kind']}")


def pipelined_worker(
    queues: WorkerQueues,
    pin: bool,
    spec: WorkerSpec,
    span_context: "SpanContext",
    early_setup: Optional[Callable] = None,
):
    """
    Main pipelined worker function.

    Parameters
    ----------

    queues
        request and response queues

    pin
        Whether or not the CPU worker should be pinned to a specific CPU

    spec
        This worker's spec, containing name, worker index, device kind etc.

    span_context
        The tracing span we should attach to for the setup code

    early_setup
        Function that will be called very early in the setup code,
        allowing to inject custom functionality or specific warmup code
    """
    # FIXME: propagate to parent process with a pipe or similar?
    sys.stderr.close()
    sys.stderr = open(os.open(os.devnull, os.O_RDWR), closefd=False)

    try:
        if early_setup:
            early_setup()

        set_file_limit()

        # FIXME: explicitly propagate exporter settings to the workers?
        # right now taken care of by environment variables...
        worker_idx = spec["worker_idx"]
        maybe_setup_tracing(service_name="pipelined_worker", service_id=f"worker-{worker_idx}")

        # attach to parent span context for startup:
        with attach_to_parent(span_context), \
                tracer.start_as_current_span("pipelined_worker.startup") as span:
            _setup_device(spec, pin)
            work_mem: dict[str, Any] = {}

            span.set_attributes({
                "libertem.spec.name": spec["name"],
                "libertem.spec.device_id": spec["device_id"],
                "libertem.spec.devide_kind": spec["device_kind"],
                "libertem.spec.worker_idx": spec["worker_idx"],
            })

            set_thread_name(f"worker-{worker_idx}")

            worker_context = PipelinedWorkerContext(queues.request, queues.message)
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
            "exception": e,
        })
        # drain, close and join queues:
        queues.request.close()
        queues.response.close(drain=False)
        queues.message.close(drain=False)
        sys.exit(1)


class PipelinedWorkerContext(WorkerContext):
    """
    A context object that is made available to the Partition
    for custom communication to its matching DataSet class
    (currently uni-directional)
    """
    def __init__(self, queue: "WorkerQueue", msg_queue: "WorkerQueue"):
        self._queue = queue
        self._msg_queue = msg_queue

    def get_worker_queue(self) -> WorkerQueue:
        return self._queue

    def signal(self, ident: str, topic: str, msg_dict: dict[str, Any]):
        msg_dict.update({'ident': ident})
        self._msg_queue.put((topic, msg_dict))


ResultT = Generator[tuple[Any, TaskProtocol], None, None]
ResultWithID = Generator[tuple[Any, TaskProtocol, int], None, None]


def _order_results(results_in: ResultWithID) -> ResultT:
    """
    Order the `results_in` generator by the result id, yielding ordered results.
    Requires indexes to be without gaps.
    """
    last_sent_id = -1
    # for results that are received out-of-order, keep sorted:
    result_stack: list[tuple[Any, TaskProtocol, int]] = []
    span = trace.get_current_span()
    for result, task, task_id in results_in:
        if task_id == last_sent_id + 1:
            span.add_event("_order_results.yield", {
                'task_id': task_id,
            })
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
            span.add_event("_order_results.yield", {
                'task_id': res[2],
            })
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


def _make_spec(
    cpus: Union[int, Iterable[int]],
    cudas: Union[int, Iterable[int]],
    has_cupy: bool = False,
) -> list[WorkerSpec]:
    """
    Takes the output of :func:`libertem.utils.devices.detect`
    and makes a plan for starting workers on them.

    Parameters
    ----------

    cpus: int | Iterable
        Iterable of integer CPU identifiers or an integer number of workers to create.
        If pinning is enabled, each worker processe is pinned to one of these identifiers,
        as accepted by :func:`python:os.sched_setaffinity`. Pinning is currently only
        supported on platforms that implement :func:`python:os.sched_setaffinity`.

    cudas: int | Iterable
        Interable of CUDA device identifiers for which workers should be started or
        an integer number of GPU workers to create across the available devices.
        Identifiers can be repeated to start multiple workers per GPU, which can
        result in better device utilization.

    has_cupy
        Is cupy available?
    """
    spec = []
    worker_idx = 0

    if isinstance(cpus, int):
        cpus = tuple(range(cpus))

    cudas = assign_cudas(cudas)

    grouped_cudas = itertools.groupby(cudas, lambda x: x)
    for device_id, group in grouped_cudas:
        for i in range(len(list(group))):
            spec.append(WorkerSpec(
                name=f"cuda-{device_id}-{i}",
                device_id=device_id,
                device_kind="CUDA",
                worker_idx=worker_idx,
                has_cupy=has_cupy,
            ))
            worker_idx += 1

    for device_id in cpus:
        spec.append(WorkerSpec(
            name=f"cpu-{device_id}",
            device_id=device_id,
            device_kind="CPU",
            worker_idx=worker_idx,
            has_cupy=False,
        ))
        worker_idx += 1

    return spec


def _raise_from_msg(msg: dict, err_prefix: str):
    if "exception" in msg:
        raise msg["exception"]
    else:
        raise RuntimeError(f"{err_prefix}: {msg['error']}")


def _inspect_startup(msg, span):
    if msg["type"] == "ERROR":
        _raise_from_msg(msg, "error on startup")
    if msg["type"] != "STARTUP_DONE":
        raise RuntimeError(
            f"unknown message type {msg['type']}, expected STARTUP_DONE"
        )
    span.add_event("worker startup done", {"worker_id": msg["worker_id"]})


class PipelinedExecutor(BaseJobExecutor):
    """
    Multi-process pipelined executor. Useful for live processing using
    `LiberTEM-live <https://libertem.github.io/LiberTEM-live/>`_
    if your processing function is not able to keep up with the incoming data
    stream in a single process, but also works for offline processing.

    Parameters
    ----------
    spec
        Specification for the worker processes - can be generated
        by :meth:`make_spec`.

    pin_workers
        Pin each CPU worker to a specific CPU, as defined by :func:`python:os.sched_setaffinity`.
        Only works on OSes that implement :func:`python:os.sched_setaffinity`.

    startup_timeout
        Startup of the executor is cancelled if it doesn't finish within
        this limit (in detail: each worker's startup time is limited by this timeout).
        In seconds.

    cleanup_timeout
        When cleaning up using :meth:`close`, give up after
        this limit. In seconds.

    early_setup
        Callable that will be run as early as possible on each worker process.
        Useful for custom warmup code or testing.

    Note
    ----
    This executor is not thread-safe - concurrent calls into :meth:`run_tasks` or
    :meth:`run_function` are not supported.
    """
    def __init__(
        self,
        spec: Optional[list[WorkerSpec]] = None,
        pin_workers: bool = True,
        startup_timeout: float = 30.0,
        cleanup_timeout: float = 10.0,
        early_setup: Optional[Callable] = None,
    ) -> None:
        self._pin_workers = pin_workers
        if spec is None:
            spec = self._default_spec()
        self._spec = spec
        self._closed = True
        self._early_setup = early_setup

        # for testing via monkeypatching, for example:
        self._worker_selector: Optional[WorkerSelector] = None

        # timeout for cleanup, either from exception or when joining processes
        self._cleanup_timeout = cleanup_timeout

        # timeout for starting a single worker
        self._startup_timeout = startup_timeout

        # keep this at the bottom:
        self._pool = self._start_pool()

    def _start_pool(self) -> WorkerPool:
        with tracer.start_as_current_span("PipelinedExecutor.start_pool") as span:
            pool = WorkerPool(
                worker_fn=functools.partial(
                    pipelined_worker,
                    pin=self._pin_workers,
                    early_setup=self._early_setup,
                ),
                spec=self._spec,
            )
            # if any processes are already dead here, raise an exception so we
            # don't have to run into a timeout below:
            pool.assert_all_alive()

            warn_time = 5.0
            check_interval = 0.1
            warn_deadline = time.monotonic() + warn_time
            startup_deadline = time.monotonic() + self._startup_timeout
            warned = False

            num_started = 0

            while num_started < pool.size:
                pool.assert_all_alive()
                try:
                    with pool.response_queue.get(timeout=check_interval) as (msg, _):
                        _inspect_startup(msg, span)
                        num_started += 1
                        continue
                except WorkerQueueEmpty:
                    if time.monotonic() > startup_deadline:
                        pool.assert_all_alive()
                        pool.kill()
                        # break possibly confusing exception chain using "from None":
                        raise RuntimeError(
                            f"Timeout while starting workers, might need to increase "
                            f"`startup_timeout` (is {self._startup_timeout}s)"
                        ) from None
                    if time.monotonic() > warn_deadline and not warned:
                        warnings.warn('Slow worker startup, please be patient...', RuntimeWarning)
                        warned = True
            for qp in pool.workers:
                qp.queues.request.put({
                    "type": "WARMUP",
                    "span_context": span.get_span_context(),
                })
            # set here, so we don't try to close the pool if it doesn't exist
            self._closed = False
            return pool

    def _restart_pool(self):
        self._pool.kill()
        self._pool = self._start_pool()

    @classmethod
    def _default_spec(cls):
        from libertem.utils.devices import detect
        detected = detect()
        return _make_spec(**detected)

    @classmethod
    def make_local(cls, **kwargs):
        """
        Create a :code:`PipelinedExecutor` with the default spec.
        """
        spec = cls._default_spec()
        return cls(spec=spec, **kwargs)

    @classmethod
    def make_spec(
        cls,
        cpus: Union[int, Iterable[int]],
        cudas: Union[int, Iterable[int]],
        has_cupy: bool = False,
    ):
        return _make_spec(cpus=cpus, cudas=cudas, has_cupy=has_cupy)
    make_spec.__doc__ = _make_spec.__doc__

    def _validate_worker_state(self):
        if not self._pool.all_alive():
            raise RuntimeError("some workers are stopped, cannot continue")

    def _run_tasks_inner(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        task_comm_handler: "TaskCommHandler",
    ) -> ResultWithID:
        # In theory, `in_flight` could be calculated from `id_to_task`, but in case
        # of exceptions, it becomes a bit harder to keep attribution of messages to
        # tasks, which is why we have a separate counter for now.
        in_flight = 0
        id_to_task: dict[int, TaskProtocol] = {}
        tasks_uuid = str(uuid.uuid4())

        all_workers = self.get_available_workers()
        scheduler = Scheduler(all_workers=all_workers)
        pool_info_for_worker = dict(zip(all_workers, self._pool.workers))
        selector: WorkerSelector

        # feature detect: can we access the queue size? support for this can be
        # missing depending on platform
        if self._worker_selector is None:
            try:
                self._pool.workers[0].queues.request.size()
                selector = _select_by_queue_size
            except NotImplementedError:
                # fall back to simple round robin worker selection
                selector = _select_by_round_robin
        else:
            selector = self._worker_selector

        try:
            self._validate_worker_state()
            task_comm_handler.start()
            span = trace.get_current_span()
            span_context = span.get_span_context()

            def yield_result_if_found(block, timeout):
                nonlocal in_flight
                try:
                    with self._pool.response_queue.get(block=block, timeout=timeout) as (result, _):
                        if result.get('uuid') != tasks_uuid:
                            # mismatch, log and ignore:
                            logger.warning(
                                "mismatched result, ignoring: %s != %s",
                                result.get('uuid'), tasks_uuid,
                            )
                            return
                        in_flight -= 1
                        if result["type"] == "ERROR":
                            _raise_from_msg(result, "failed to run tasks")
                        result_task_id = result["task_id"]
                        yield (result["result"], id_to_task[result_task_id], result_task_id)
                        del id_to_task[result_task_id]
                        if len(id_to_task) != in_flight:
                            raise RuntimeError(
                                "state mismatch; `id_to_task` mapping should match `in_flight`"
                            )
                except WorkerQueueEmpty:
                    self._validate_worker_state()

            with tracer.start_as_current_span('schedule_tasks') as span:
                for task_idx, task in enumerate(tasks):
                    # important: call `schedule_task` before incrementing number of
                    # in-flight tasks, otherwise, if we run into an exception, we
                    # will wait for a response that never comes.
                    # select a worker for `task` according to `selector`:
                    _worker_idx, worker_queues = schedule_task(
                        task_idx,
                        task,
                        self._pool,
                        scheduler,
                        pool_info_for_worker,
                        selector=selector,
                    )
                    worker_queues.request.put({
                        "type": "RUN_TASK",
                        "uuid": tasks_uuid,
                        "task": cloudpickle.dumps(task),
                        "task_id": task_idx,
                        "params_handle": params_handle,
                        "span_context": span_context,
                    })
                    span.add_event('RUN_TASK', {
                        'task_id': task_idx,
                    })
                    in_flight += 1
                    id_to_task[task_idx] = task
                    if len(id_to_task) != in_flight:
                        raise RuntimeError(
                            "state mismatch; `id_to_task` mapping should match `in_flight`"
                        )

                    # FIXME: semantics of this - is this enough?
                    # does it matter if this is enough? we can change it in the future if not
                    # could be: the function returns once it has forwarded
                    # all the data necessary for the given task,
                    # (or, in the offline case, immediately)
                    try:
                        task_comm_handler.handle_task(task, worker_queues.request)
                    except JobCancelledError:
                        worker_queues.request.put({
                            "type": "END_TASK",
                            "task_id": task_idx,
                            "params_handle": params_handle,
                            "span_context": span_context,
                        })
                        raise

                    # NOTE: sentinel message; in case of errors, the worker
                    # needs to discard the data from the queue until it receives
                    # this message:
                    worker_queues.request.put({
                        "type": "END_TASK",
                        "task_id": task_idx,
                        "params_handle": params_handle,
                        "span_context": span_context,
                    })

                    yield from yield_result_if_found(block=False, timeout=None)

            # at the end, block to get the remaining results:
            while in_flight > 0:
                yield from yield_result_if_found(block=True, timeout=0.1)
        except JobCancelledError:
            # don't drain here, as the next acquisition could start
            # very soon; instead, we just ignore the mismatched
            # responses in the next run
            raise
        except Exception as e:
            # In case of an exception, we need to drain the response queue,
            # so the next `run_tasks` call isn't polluted by old responses.
            # -> just like in the happy case, we need to wait for responses
            # for all in-flight tasks. In case the error happened between incrementing
            # `in_flight` and actually sending the task to the queue, we should
            # have a timeout here to not wait infinitely long.
            try:
                self._drain_response_queue(in_flight=in_flight)
            except RuntimeError as e2:
                raise e2 from e
            # if from a worker, this is the first exception that got put into the queue
            raise
        finally:
            task_comm_handler.done()

    def _drain_response_queue(self, in_flight: int) -> None:
        """
        Drain response queue and log any errors returned from workers

        Parameters
        ----------
        in_flight : int
            The number of requests that are still in flight
        """
        t0 = time.time()
        while in_flight > 0:
            try:
                timeout = max(0.010, self._cleanup_timeout - (time.time() - t0))
                with self._pool.response_queue.get(
                    timeout=timeout,
                ) as (result, _):
                    t0 = time.time()
                    in_flight -= 1
                    # we only raise the first exception; log the others here:
                    if result["type"] == "ERROR":
                        logger.error(f"Error response from worker: {result['error']}")
            except WorkerQueueEmpty:
                # kill and restart workers:
                self._restart_pool()
                raise RuntimeError(
                    f'Worker or queue presumably in a bad state, lost {in_flight} in-flight tasks.'
                )

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        task_comm_handler: "TaskCommHandler",
    ) -> ResultT:
        with tracer.start_as_current_span("PipelinedExecutor.run_tasks"):
            with task_comm_handler.monitor(self._pool.message_queue):
                yield from _order_results(self._run_tasks_inner(
                    tasks, params_handle, cancel_id, task_comm_handler,
                ))

    def get_available_workers(self) -> WorkerSet:
        return WorkerSet([
            Worker(
                name=worker_info.spec["name"],
                host="localhost",
                resources=_resources_for_spec(worker_info.spec),
                nthreads=1,
            )
            for worker_info in self._pool.workers
        ])

    def close(self):
        with tracer.start_as_current_span("PipelinedExecutor.close") as span:
            if self._closed:
                return
            for idx, worker_info in enumerate(self._pool.workers):
                span.add_event("sending SHUTDOWN", {"idx": idx})
                worker_info.queues.request.put({
                    "type": "SHUTDOWN",
                    "span_context": span.get_span_context(),
                })
                span.add_event("SHUTDOWN sent", {"idx": idx})
                while True:
                    try:
                        with worker_info.queues.response.get(block=False) as msg:
                            logger.warning(f"got message on close: {msg[0]}")
                    except WorkerQueueEmpty:
                        break
                worker_info.process.join(timeout=self._cleanup_timeout)
                if worker_info.process.exitcode is None:
                    self._pool.kill_worker(worker_info)
                worker_info.queues.request.close(force=True)
            self._pool.close_resp_queue()
            self._pool.close_mesg_queue()
            self._closed = True

    def __del__(self):
        # `self` may be already partially garbage-collected; only close
        # if "enough" of `self` still exists:
        if hasattr(self, '_closed') and not self._closed:
            self.close()

    def _run_function(self, fn: Callable[..., T], worker_idx, *args, **kwargs) -> T:
        self._validate_worker_state()
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
            if response["type"] == "ERROR":
                if "exception" in response:
                    raise response["exception"]
                else:
                    raise RuntimeError(f"failed to run function: {response['error']}")
            if not response["type"] == "RUN_FUNCTION_RESULT":
                raise RuntimeError(f"invalid response type: {response['type']}")
            result: T = response["result"]
            return result

    def run_function(self, fn: Callable[..., T], *args, **kwargs) -> T:
        # FIXME: this is not concurrency-safe currently! beware!
        with tracer.start_as_current_span("PipelinedExecutor.run_function"):
            return self._run_function(fn, 0, *args, **kwargs)

    def run_each_worker(self, fn, *args, **kwargs):
        # FIXME: not as fast as it could be, but also not really perf-sensitive?
        result = {}
        for idx, worker_info in enumerate(self._pool.workers):
            result[worker_info.spec["name"]] = self._run_function(
                fn, worker_idx=idx, *args, **kwargs
            )
        return result

    def run_each_host(self, fn, *args, **kwargs):
        return {"localhost": self.run_function(fn, *args, **kwargs)}

    @contextlib.contextmanager
    def scatter(self, obj):
        self._validate_worker_state()
        key = str(uuid.uuid4())
        with tracer.start_as_current_span("scatter_put_to_workers"):
            for worker_info in self._pool.workers:
                worker_info.queues.request.put({
                    "type": "SCATTER",
                    "key": key,
                    "value": obj,
                })
        try:
            yield key
        finally:
            if not self._closed:
                for worker_info in self._pool.workers:
                    worker_info.queues.request.put({
                        "type": "DELETE",
                        "key": key,
                    })

    def scatter_update(self, handle: str, obj):
        self._validate_worker_state()
        for worker_info in self._pool.workers:
            worker_info.queues.request.put({
                "type": "SCATTER_UPDATE",
                "key": handle,
                "value": obj,
            })

    def scatter_update_patch(self, handle: str, patch):
        self._validate_worker_state()
        for worker_info in self._pool.workers:
            worker_info.queues.request.put({
                "type": "SCATTER_UPDATE_PATCH",
                "key": handle,
                "patch": patch,
            })

    def map(self, fn, iterable):
        # FIXME: replace with efficient impl if needed
        with tracer.start_as_current_span("PipelinedExecutor.map"):
            return [
                self._run_function(fn=lambda: fn(item), worker_idx=0)
                for item in iterable
            ]
