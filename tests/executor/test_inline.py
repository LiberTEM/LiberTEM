from libertem.executor.inline import InlineJobExecutor


def test_run_each_worker_1():
    def fn1():
        return "some result"

    executor = InlineJobExecutor()

    results = executor.run_each_worker(fn1)
    assert len(results.keys()) == 1
    assert len(results.keys()) == len(executor.get_available_workers())

    k = next(iter(results))
    result0 = results[k]
    assert result0 == "some result"
    assert k == "inline"

