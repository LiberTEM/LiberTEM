import os

import pytest


@pytest.mark.dist
def test_sum_on_dist(raw_on_workers_ipy, ipy_ctx):
    print(ipy_ctx.executor.run_each_host(lambda: os.system("hostname")))
    print(ipy_ctx.executor.get_available_workers().group_by_host())
    print(ipy_ctx.executor.get_available_workers())
    print(ipy_ctx.executor.run_each_host(
        lambda: os.listdir(os.path.dirname(raw_on_workers_ipy._path))))
    analysis = ipy_ctx.create_sum_analysis(dataset=raw_on_workers_ipy)
    results = ipy_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
