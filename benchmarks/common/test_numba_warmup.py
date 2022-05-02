import pytest

from libertem.web.dataset import prime_numba_cache


@pytest.mark.benchmark()
def test_numba_prime_cache(lt_ctx, benchmark, large_raw):
    """
    Primes the cache for the raw dataset. this only makes sense to be run individually, really,
    or we need to find a way to dynamically set the numba cache path somehow, and/or run the
    test in a subprocess.
    """
    benchmark.pedantic(prime_numba_cache, kwargs=dict(ds=large_raw), rounds=5, warmup_rounds=0)
