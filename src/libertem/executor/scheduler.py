from collections import defaultdict


class WorkerSet:
    def __init__(self, workers):
        self.workers = workers

    def group_by_host(self):
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

    def get_by_host(self, host):
        return self.filter(lambda w: w.host == host)

    def example(self):
        """
        get a single representative from this WorkerSet
        """
        return self.workers[0]

    def filter(self, fn):
        return WorkerSet([
            w
            for w in self.workers
            if fn(w)
        ])

    def has_cpu(self):
        return self.filter(lambda worker: bool(worker.resources.get('CPU', False)))

    def has_cuda(self):
        return self.filter(lambda worker: bool(worker.resources.get('CUDA', False)))

    def has_threaded_workers(self):
        return any(w.nthreads != 1 for w in self.workers)

    def concurrency(self):
        return sum(w.nthreads for w in self.workers)

    def hosts(self):
        return {worker.host for worker in self.workers}

    def names(self):
        return [worker.name for worker in self.workers]

    def extend(self, other):
        return WorkerSet(self.workers + other.workers)

    def __iter__(self):
        return iter(self.workers)

    def __len__(self):
        return len(self.workers)

    def __repr__(self):
        return "<WorkerSet {}>".format(
            self.workers,
        )

    def __eq__(self, other):
        return self.workers == other.workers


class Worker:
    """
    A reference to a worker process identified by `name` running on `host`.
    """
    def __init__(self, name, host, resources, nthreads):
        self.name = name
        self.host = host
        self.resources = resources
        self.nthreads = nthreads

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<Worker {self.name} on {self.host} with {self.resources}, {self.nthreads} threads>"

    def __eq__(self, other):
        return self.name == other.name and self.host == other.host


class Scheduler:
    def __init__(self, all_workers: WorkerSet):
        self.workers = all_workers

    def workers_for_task(self, task):
        """
        Given a task, return a WorkerSet
        """
        raise NotImplementedError()

    def workers_for_partition(self, partition):
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
