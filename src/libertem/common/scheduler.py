from collections import defaultdict
from typing import List, Set, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .executor import TaskProtocol, ResourceDef
    from libertem.io.dataset.base import Partition


class WorkerSet:
    def __init__(self, workers: List["Worker"]):
        self.workers = workers

    def group_by_host(self) -> List["WorkerSet"]:
        """
        returns a list of `WorkerSet`s, each containing the workers for a single host
        """
        by_host = defaultdict(lambda: [])
        for worker in self.workers:
            by_host[worker.host].append(worker)

        result = []
        for host, workers in by_host.items():
            result.append(WorkerSet(workers))
        return result

    def get_by_host(self, host) -> "WorkerSet":
        return self.filter(lambda w: w.host == host)

    def example(self):
        """
        get a single representative from this WorkerSet
        """
        return self.workers[0]

    def filter(self, fn) -> "WorkerSet":
        return WorkerSet([
            w
            for w in self.workers
            if fn(w)
        ])

    def has_cpu(self) -> "WorkerSet":
        return self.filter(lambda worker: bool(worker.resources.get('CPU', False)))

    def has_cuda(self) -> "WorkerSet":
        return self.filter(lambda worker: bool(worker.resources.get('CUDA', False)))

    def has_threaded_workers(self) -> bool:
        return any(w.nthreads != 1 for w in self.workers)

    def concurrency(self) -> int:
        return sum(w.nthreads for w in self.workers)

    def hosts(self) -> Set[str]:
        return {worker.host for worker in self.workers}

    def names(self) -> List[str]:
        return [worker.name for worker in self.workers]

    def extend(self, other) -> "WorkerSet":
        return WorkerSet(self.workers + other.workers)

    def __iter__(self) -> Iterator["Worker"]:
        return iter(self.workers)

    def __len__(self) -> int:
        return len(self.workers)

    def __repr__(self) -> str:
        return "<WorkerSet {}>".format(
            self.workers,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkerSet):
            return False
        return self.workers == other.workers


class Worker:
    """
    A reference to a worker process identified by `name` running on `host`.
    """
    def __init__(
        self,
        name: str,
        host: str,
        resources: "ResourceDef",
        nthreads: int
    ):
        self.name = name
        self.host = host
        self.resources = resources
        self.nthreads = nthreads

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<Worker {self.name} on {self.host} with {self.resources}, {self.nthreads} threads>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Worker):
            return False
        return self.name == other.name and self.host == other.host


class Scheduler:
    def __init__(self, all_workers: WorkerSet):
        self.workers = all_workers

    def workers_for_task(self, task: "TaskProtocol"):
        """
        Given a task, return a WorkerSet
        """
        raise NotImplementedError()

    def workers_for_partition(self, partition: "Partition"):
        """
        Given a partition, return a WorkerSet
        """
        pass

    def effective_worker_count(self):
        '''
        Return the effective number of workers for partitioning

        This avoids residual partitions that would degrade performance.
        '''
        cpu_workers = self.workers.has_cpu()
        gpu_workers = self.workers.has_cuda()

        # Mixed case: return only CPU workers or GPU workers, whichever is
        # larger, to not have residual partitions in CPU-only or GPU-only
        # processing. Plus, a GPU worker will spin a CPU at 100 % while running.

        if cpu_workers:
            return max(cpu_workers.concurrency(), gpu_workers.concurrency())
        # GPU-only
        elif gpu_workers:
            return gpu_workers.concurrency()
        else:
            return self.workers.concurrency()
