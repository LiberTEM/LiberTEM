from libertem.executor.dask import CommonDaskMixin


def test_task_affinity_1():
    cdm = CommonDaskMixin()
    workers = [
        {'host': '127.0.0.1', 'name': 'w1'},
        {'host': '127.0.0.1', 'name': 'w2'},
        {'host': '127.0.0.1', 'name': 'w3'},
        {'host': '127.0.0.1', 'name': 'w4'},

        {'host': '127.0.0.2', 'name': 'w5'},
        {'host': '127.0.0.2', 'name': 'w6'},
        {'host': '127.0.0.2', 'name': 'w7'},
        {'host': '127.0.0.2', 'name': 'w8'},
    ]

    assert cdm._task_idx_to_workers(workers, 0) == ['w1', 'w2', 'w3', 'w4']
    assert cdm._task_idx_to_workers(workers, 1) == ['w5', 'w6', 'w7', 'w8']
    assert cdm._task_idx_to_workers(workers, 2) == ['w1', 'w2', 'w3', 'w4']
    assert cdm._task_idx_to_workers(workers, 3) == ['w5', 'w6', 'w7', 'w8']
