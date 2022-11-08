DEFAULT_RAM_PER_WORKER = 4*1024*1024*1024


def make_gpu_plan(cudas, cuda_info, max_workers_per_cuda, ram_per_cuda_worker):
    gpu_plan = []
    disposable_ram = {}
    worker_count = {}

    if isinstance(cudas, int):
        cudas = tuple(range(cudas))

    for cuda in cudas:
        disposable_ram[cuda] = cuda_info[cuda]['mem_info'][1]
        worker_count[cuda] = 0
    for cuda in cudas:
        while (
                disposable_ram[cuda] >= ram_per_cuda_worker
                and worker_count[cuda] < max_workers_per_cuda
        ):
            gpu_plan.append((cuda, worker_count[cuda]))
            worker_count[cuda] += 1
            disposable_ram[cuda] -= ram_per_cuda_worker

    # order by worker index so that we first create one worker on each
    # GPU instead of filling up the first GPU.
    # In case we are short on CPU cores for the GPU workers,
    # at least we try to cover all GPUs with some workers.
    gpu_plan.sort(key=lambda x: x[1])
    return gpu_plan
