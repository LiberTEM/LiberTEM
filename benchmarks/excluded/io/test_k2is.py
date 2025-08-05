import os
import glob

import pytest
import numpy as np

from libertem import api
from libertem.udf.masks import ApplyMasksUDF

from utils import drop_cache, warmup_cache, get_testdata_prefixes, backends_by_name


K2IS_FILE = "K2IS/Capture52/Capture52_.gtg"

PREFIXES = get_testdata_prefixes()


def filelist(gtg_file):
    root, ext = os.path.splitext(gtg_file)
    return glob.glob(root + "*")


@pytest.mark.benchmark(
    group="io"
)
@pytest.mark.parametrize(
    "drop", ("cold_cache", "warm_cache")
)
@pytest.mark.parametrize(
    "prefix", PREFIXES
)
def test_sequential(benchmark, prefix, drop):
    hdr = os.path.join(prefix, K2IS_FILE)

    flist = filelist(hdr)

    if drop == "cold_cache":
        drop_cache(flist)
    elif drop == "warm_cache":
        warmup_cache(flist)
    else:
        raise ValueError("bad param")

    benchmark.pedantic(
        warmup_cache, args=(flist, ),
        warmup_rounds=0,
        rounds=1,
        iterations=1
    )


class TestUseSharedExecutor:
    @pytest.mark.benchmark(
        group="io"
    )
    @pytest.mark.parametrize(
        "drop", ("cold_cache", "warm_cache")
    )
    @pytest.mark.parametrize(
        "prefix", PREFIXES
    )
    @pytest.mark.parametrize(
        "io_backend", ("mmap", "mmap_readahead", "buffered", "direct"),
    )
    def test_mask(self, benchmark, prefix, drop, shared_dist_ctx, io_backend):
        io_backend = backends_by_name[io_backend]
        hdr = os.path.join(prefix, K2IS_FILE)
        flist = filelist(hdr)

        ctx = shared_dist_ctx
        ds = ctx.load(filetype="k2is", path=hdr, io_backend=io_backend)

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        # warmup executor
        ctx.run_udf(udf=udf, dataset=ds)

        if drop == "cold_cache":
            drop_cache(flist)
        elif drop == "warm_cache":
            warmup_cache(flist)
        else:
            raise ValueError("bad param")

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=1,
            iterations=1,
        )

    @pytest.mark.parametrize(
        "prefix", PREFIXES
    )
    @pytest.mark.benchmark(
        group="io"
    )
    def test_mask_repeated(self, benchmark, prefix, lt_ctx):
        hdr = os.path.join(prefix, K2IS_FILE)
        flist = filelist(hdr)

        ctx = lt_ctx
        ds = ctx.load(filetype="k2is", path=hdr)

        sig_shape = ds.shape.sig

        def mask():
            return np.ones(sig_shape, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        # warmup:
        ctx.run_udf(udf=udf, dataset=ds)
        warmup_cache(flist)

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=3,
            iterations=1,
        )


@pytest.mark.benchmark(
    group="io"
)
@pytest.mark.parametrize(
    "first", ("warm_executor", "cold_executor", )
)
@pytest.mark.parametrize(
    "prefix", PREFIXES[:1]
)
@pytest.mark.parametrize(
    "io_backend", ("mmap", "mmap_readahead", "buffered"),
)
def test_mask_firstrun(benchmark, prefix, first, io_backend):
    io_backend = backends_by_name[io_backend]
    hdr = os.path.join(prefix, K2IS_FILE)
    flist = filelist(hdr)

    with api.Context() as ctx:
        ds = ctx.load(filetype="k2is", path=hdr, io_backend=io_backend)

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        if first == "warm_executor":
            ctx.run_udf(udf=udf, dataset=ds)
        elif first == "cold_executor":
            pass
        else:
            raise ValueError("bad param")

        warmup_cache(flist)

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=1,
            iterations=1
        )
