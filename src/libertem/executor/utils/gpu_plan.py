from typing import Iterable, Union
import itertools
import warnings


DEFAULT_RAM_PER_WORKER = 4*1024*1024*1024


def assign_cudas(cudas: Union[int, Iterable[int]]) -> Iterable[int]:
    """
    Takes the cudas argument to :code:`cluster_spec` and
    converts it into a properly formatted iterable of CUDA
    device ids

    If cudas is an integer, assigns this many workers to
    device ids in a round-robin fashion, where CUDA devices
    can be detected. If the devices cannot be detected raise
    a warning and assign device_ids in a sequential fashion.

    Will also raise a warning if cudas is a non-empty iterable
    on a system where the CUDA devices cannot be detected.
    """
    if isinstance(cudas, int) or len(cudas):
        # Needed to know if we can assign CUDA workers
        from libertem.utils.devices import detect
        avail_cudas = detect()['cudas']
        if not avail_cudas and cudas:  # needed in case cudas == 0
            warnings.warn('Specifying CUDA workers on system with '
                          'no visible CUDA devices',
                          RuntimeWarning)
            # If we are assigning from int, just use increasing
            # device indices even if they are unavailable
            avail_cudas = itertools.count()

        if isinstance(cudas, int):
            # Round-Robin-assign to available CUDA devices
            # Can override by specifying cudas as an iterable
            cudas_iter = itertools.cycle(avail_cudas)
            cudas = tuple(next(cudas_iter) for _ in range(cudas))

    return cudas


def make_gpu_plan(cudas, cuda_info, max_workers_per_cuda, ram_per_cuda_worker):
    gpu_plan = []
    disposable_ram = {}
    worker_count = {}

    # For backwards compatibility we hallucinate specs
    # for a low-end GPU like T500 that can support one typical GPU worker.
    # In Feb 2023 low-end NVidia GPUs have at least 6 GB RAM
    if cuda_info is None:
        cuda_info = {}
        for cuda in cudas:
            cuda_info[cuda] = {}
            cuda_info[cuda]['mem_info'] = (DEFAULT_RAM_PER_WORKER, DEFAULT_RAM_PER_WORKER)

    for cuda in cudas:
        disposable_ram[cuda] = cuda_info[cuda]['mem_info'][1]
        worker_count[cuda] = 0
    for cuda in cudas:
        # We assign at least one worker for each time a GPU
        # ID is listed in cudas
        while (True):
            gpu_plan.append((cuda, worker_count[cuda]))
            worker_count[cuda] += 1
            disposable_ram[cuda] -= ram_per_cuda_worker
            if disposable_ram[cuda] < ram_per_cuda_worker:
                break
            if worker_count[cuda] >= max_workers_per_cuda:
                break

    # order by worker index so that we first create one worker on each
    # GPU instead of filling up the first GPU.
    # In case we are short on CPU cores for the GPU workers,
    # at least we try to cover all GPUs with some workers.
    gpu_plan.sort(key=lambda x: x[1])
    return gpu_plan
