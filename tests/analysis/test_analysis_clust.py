import numpy as np

import pytest

from libertem.analysis.clust import ClusterAnalysis
from libertem.io.dataset.memory import MemoryDataSet
from libertem.executor.base import AsyncAdapter


@pytest.mark.asyncio
@pytest.mark.slow
async def test_cluster_analysis(inline_executor):
    data = np.zeros([16, 16, 8, 8]).astype(np.float32)
    data[:, 2, 2] = 7
    # adding strong non-zero order diffraction peaks for 0:3 frames
    data[0:3, 0, 0] = 2
    data[0:3, 4, 4] = 2
    # adding weak non-zero order diffraction peaks for 0:3 frames
    data[3:6, 2, 0] = 1
    data[3:6, 2, 4] = 1

    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=1, sig_dims=2)

    executor = AsyncAdapter(wrapped=inline_executor)

    analysis = ClusterAnalysis(dataset=dataset, parameters={
        'n_peaks': 23,
        'n_clust': 7,
        'cy': 3,
        'cx': 3,
        'ri': 1,
        'ro': 5,
        'delta': 0.05,
        'min_dist': 1,
    })

    uuid = 'bd3b39fb-0b34-4a45-9955-339da6501bbb'

    async def send_results(results, finished):
        pass

    await analysis.controller(
        cancel_id=uuid, executor=executor,
        job_is_cancelled=lambda: False,
        send_results=send_results,
    )
