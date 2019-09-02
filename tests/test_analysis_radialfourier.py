import pytest

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


@pytest.fixture
def ds_random():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        num_partitions=2,
    )
    return dataset


def test_smoke(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, ri=0, ro=10, n_bins=2, max_order=23
    )
    results = lt_ctx.run(analysis)


def test_smoke_small(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, ri=0, ro=1, n_bins=1, max_order=23
    )
    results = lt_ctx.run(analysis)
    # access result to actually create result list:
    results['absolute_0_0']


def test_smoke_large(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, n_bins=1, max_order=23
    )
    results = lt_ctx.run(analysis)


def test_smoke_defaults(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random
    )
    results = lt_ctx.run(analysis)


def test_sparse(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, use_sparse=True,
    )
    results = lt_ctx.run(analysis)
