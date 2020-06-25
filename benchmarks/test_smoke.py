import pytest
import numba


@numba.njit
def hello():
    return "world"


@pytest.mark.compilation
@pytest.mark.benchmark(
    group="compilation"
)
def test_numba_compilation(benchmark):
    benchmark.extra_info["mark"] = "compilation"
    benchmark.pedantic(hello, warmup_rounds=0, rounds=2, iterations=1)


@pytest.mark.benchmark(
    group="smoke"
)
def test_numba_performance(benchmark):
    benchmark(hello)
