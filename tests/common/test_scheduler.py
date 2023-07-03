from libertem.common.scheduler import Worker, WorkerSet, Scheduler
from libertem.common.executor import ResourceDef


class MockTask:
    def __init__(self, resources: ResourceDef):
        self._resources = resources

    def get_resources(self) -> ResourceDef:
        return self._resources


def test_workers_for_task():
    cpu_resources: ResourceDef = {
        "compute": 1,
        "ndarray": 1,
        "CPU": 1,
    }
    cupy_resources: ResourceDef = {
        "compute": 1,
        "ndarray": 1,
        "CUDA": 1,
    }
    cuda_resources: ResourceDef = {
        "compute": 1,
        "CUDA": 1,
    }
    ws_cupy = WorkerSet([
        Worker(host='127.0.0.1', name='w_cupy', resources=cupy_resources, nthreads=1),
    ])
    ws_cuda = WorkerSet([
        Worker(host='127.0.0.1', name='w_cuda', resources=cuda_resources, nthreads=1),
    ])
    ws_cuda = ws_cuda.extend(ws_cupy)
    ws_numpy = WorkerSet([
        Worker(host='127.0.0.1', name='w_numpy', resources=cpu_resources, nthreads=1),
    ])
    ws = WorkerSet([
        Worker(host='127.0.0.1', name='w_empty', resources={}, nthreads=1),
    ])
    ws = ws.extend(ws_cuda)
    ws = ws.extend(ws_numpy)
    scheduler = Scheduler(all_workers=ws)

    task_cuda = MockTask(
        resources={'compute': 1, 'CUDA': 1}
    )
    task_cupy = MockTask(
        resources={'compute': 1, 'CUDA': 1, 'ndarray': 1}
    )
    task_numpy = MockTask(
        resources={'compute': 1, 'CPU': 1, 'ndarray': 1}
    )

    cuda_workers = scheduler.workers_for_task(task=task_cuda)
    assert len(cuda_workers) == 2
    assert cuda_workers == ws_cuda

    cupy_workers = scheduler.workers_for_task(task=task_cupy)
    assert len(cupy_workers) == 1
    assert cupy_workers == cupy_workers

    numpy_workers = scheduler.workers_for_task(task=task_numpy)
    assert len(numpy_workers) == 1
    assert numpy_workers == numpy_workers


def test_request_too_much():
    cpu_resources: ResourceDef = {
        "compute": 1,
        "ndarray": 1,
        "CPU": 1,
    }
    ws = WorkerSet([
        Worker(host='127.0.0.1', name='w_numpy', resources=cpu_resources, nthreads=1),
    ])
    scheduler = Scheduler(all_workers=ws)
    # the task requests a too large amount of the 'compute' resource:
    task_numpy = MockTask(
        resources={'compute': 2, 'CPU': 1, 'ndarray': 1}
    )
    numpy_workers = scheduler.workers_for_task(task=task_numpy)
    assert len(numpy_workers) == 0


def test_invalid_compares():
    assert not WorkerSet([]) == []
    assert not Worker(host='127.0.0.1', name='w_empty', resources={}, nthreads=1) == []
