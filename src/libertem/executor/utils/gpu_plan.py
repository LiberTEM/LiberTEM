from typing import Iterable, Union, Tuple, Optional
import itertools
import warnings

from libertem.utils.devices import detect

DEFAULT_RAM_PER_WORKER = 4*1024*1024*1024


class ResourceSpec:
    def __init__(self, cpus: Tuple[int], cudas: Tuple[int], has_cupy: bool, hybrid_workers: bool):
        self.cpus = tuple(cpus)
        self.cudas = tuple(cudas)
        self.has_cupy = has_cupy
        self.hybrid_workers = hybrid_workers

    @classmethod
    def create(cls,
               cpus: Optional[Union[int, Iterable[int]]] = None,
               cudas: Optional[Union[int, Iterable[int]]] = None,
               max_workers_per_cuda: Optional[int] = None,
               ram_per_cuda_worker: Optional[Union[int, float]] = None,
               ram_per_cpu_worker: Optional[Union[int, float]] = None,
               hybrid_workers: bool = True):
        detected = detect()

        if max_workers_per_cuda is None:
            max_workers_per_cuda = 4

        if ram_per_cuda_worker is None:
            ram_per_cuda_worker = DEFAULT_RAM_PER_WORKER

        if ram_per_cpu_worker is None:
            ram_per_cpu_worker = DEFAULT_RAM_PER_WORKER
        if cpus is None:
            max_workers = max(1, int(detected['meminfo']['total'] / ram_per_cpu_worker))
            cpus = detected['cpus'][:max_workers]
        elif isinstance(cpus, int):
            if cpus > len(detected['cpus']):
                warnings.warn(
                    "More CPU workers requested than physical cores available. "
                    "This may degrade performance."
                )
            real_cpus = []
            # If less CPUs requested than available this takes one round
            # and slices out as many as requested from the available ones.
            # If more CPUs are requested than available it repeats the available CPUs and slices
            # off the rest at the end of the last repetition
            while cpus > 0:
                cpu_slice = detected['cpus'][:cpus]
                real_cpus += cpu_slice
                cpus -= len(cpu_slice)
            # Appended as many as requested
            assert cpus == 0
            cpus = real_cpus

        actual_ram_per_cpu = detected['meminfo']['total'] / len(cpus)
        if actual_ram_per_cpu < ram_per_cpu_worker:
            warnings.warn(
                f"RAM per CPU worker is {actual_ram_per_cpu}, "
                f"less than limit {ram_per_cpu_worker}."
            )

        if isinstance(cudas, int):
            cuda_count = cudas
        elif cudas is None:
            cuda_count = None

        if cudas is None or isinstance(cudas, int):
            ram_fraction_per_worker = {}
            worker_id = {}
            cuda_ids = detected['cudas']
            available_ram_fraction = {}
            for cuda in cuda_ids:
                available_ram_fraction[cuda] = 1.
                cuda_ram = detected['cuda_info'][cuda]['mem_info'][1]
                ram_fraction_per_worker[cuda] = ram_per_cuda_worker / cuda_ram
                worker_id[cuda] = 0

            native_cuda_count = sum(
                min(max_workers_per_cuda, int(1/w))
                for w in ram_fraction_per_worker.values()
            )
            min_cudas = 0 if cuda_count is None else cuda_count

            gpu_plan = []

            real_cuda_count = max(min_cudas, native_cuda_count)

            while real_cuda_count > 0:
                all_maxed = all(n >= max_workers_per_cuda for n in worker_id.values())
                # Select the device ID with highest available RAM fraction
                # Exclude devices that have the maximum number of workers
                # Start filling beyond that limit if all devices are maxed
                sorted_by_workers = sorted(
                    (
                        (key, fraction) for (key, fraction) in available_ram_fraction.items()
                        if all_maxed or worker_id[key] < max_workers_per_cuda
                    ),
                    key=lambda x: x[1], reverse=True
                )
                selected_id = sorted_by_workers[0][0]
                # Append to plan and keep track which worker ID this is on the device
                gpu_plan.append((selected_id, worker_id[selected_id]))
                # Consume RAM fraction of worker and increase worker ID
                available_ram_fraction[selected_id] -= ram_fraction_per_worker[selected_id]
                worker_id[selected_id] += 1
                real_cuda_count -= 1

            # We SHOULD have exhausted all the available workers in the default case
            # and not exceeded the worker count limit
            if cuda_count is None:
                assert all(
                    (  # Exhaust RAM or worker count limit
                        (fraction < ram_fraction_per_worker[cuda])
                        or (worker_id[cuda] >= max_workers_per_cuda)
                    )
                    # Don't assign beyond worker count limit
                    and worker_id[cuda] <= max_workers_per_cuda
                    for cuda, fraction in ram_fraction_per_worker.values()
                )

            oversubscribed = [
                (cuda, f"{-fraction * 100}%")
                for (cuda, fraction) in ram_fraction_per_worker.items() if fraction < 0
            ]

            if cuda_count is None:
                assert not oversubscribed

            # Sort by worker ID per GPU so that we first cover all available GPUs
            # in case the worker count is restricted below the native occupancy
            in_order = sorted(gpu_plan, key=lambda x: x[1])

            if cuda_count is not None:
                # The upstream code should make sure that we have enough entries
                assert len(in_order) >= cuda_count
                in_order = in_order[:cuda_count]

            cudas = [item[0] for item in in_order]

        ram_per_cuda = {}
        for cuda in detected['cudas']:
            ram_per_cuda[cuda] = detected['cuda_info'][cuda]['mem_info'][1]

        for cuda in cudas:
            if cuda not in ram_per_cuda:
                warnings.warn(f'Specified CUDA ID {cuda} not detected in system.')
            ram_per_cuda[cuda] -= ram_per_cuda_worker

        undersubscribed = [
            (cuda, ram) for cuda, ram in ram_per_cuda.items() if ram >= ram_per_cuda_worker
        ]
        oversubscribed = [(cuda, ram) for cuda, ram in ram_per_cuda.items() if ram < 0]

        if undersubscribed:
            warnings.warn(f"Undersubscribed CUDA devices: {undersubscribed}")

        if oversubscribed:
            warnings.warn(f"Oversubscribed CUDA devices: {oversubscribed}")

        return cls(
            cpus=cpus,
            cudas=cudas,
            has_cupy=detected['has_cupy'],
            hybrid_workers=hybrid_workers
        )


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
