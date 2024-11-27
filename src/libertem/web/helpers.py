from typing import Any

from libertem.executor.base import AsyncAdapter
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from libertem.utils.devices import detect


def _int_or_zero(value) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _convert_device_map(raw_cudas: dict[int, Any]) -> list[int]:
    return [
        this_id
        for dev_id, num in raw_cudas.items()
        for this_id in [dev_id]*_int_or_zero(num)
    ]


def create_executor(*, connection, local_directory, preload, snooze_timeout) -> DaskJobExecutor:
    devices = detect()
    options = {
        "local_directory": local_directory,
    }
    if "numWorkers" in connection:
        num_workers = connection["numWorkers"]
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError('Number of workers must be positive integer')
        devices["cpus"] = range(num_workers)
    raw_cudas = connection.get("cudas", {})
    cudas = _convert_device_map(raw_cudas)
    devices["cudas"] = cudas
    return DaskJobExecutor.make_local(
        spec=cluster_spec(
            **devices,
            options=options,
            preload=preload,
        ),
        snooze_timeout=snooze_timeout,
    )


def create_executor_external(
    executor_spec: dict[str, int],
    local_directory,
    preload,
    snooze_timeout,
) -> tuple[AsyncAdapter, dict[str, dict[str, Any]]]:
    cudas = {}
    if executor_spec['cudas']:
        cudas[0] = executor_spec['cudas']
    params = {
        "connection": {
            "type": "LOCAL",
            "numWorkers": executor_spec['cpus'],
            "cudas": cudas,
        }
    }
    sync_executor = create_executor(
        connection=params['connection'],
        local_directory=local_directory,
        preload=preload,
        snooze_timeout=snooze_timeout,
    )
    pool = AsyncAdapter.make_pool()
    executor = AsyncAdapter(wrapped=sync_executor, pool=pool)
    return executor, params
