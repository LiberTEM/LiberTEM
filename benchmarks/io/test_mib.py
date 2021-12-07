import os

import pytest
import numpy as np

from libertem import api
from libertem.udf.masks import ApplyMasksUDF

from utils import drop_cache, warmup_cache, get_testdata_prefixes, backends_by_name


MIB_FILE = "MIB/20200518 165148/default.hdr"

PREFIXES = get_testdata_prefixes()


def filelist(mib_hdr):
    mib_dir = os.path.dirname(mib_hdr)
    return [os.path.join(mib_dir, fname) for fname in os.listdir(mib_dir)]


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
    mib_hdr = os.path.join(prefix, MIB_FILE)

    flist = filelist(mib_hdr)

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
    @pytest.mark.parametrize(
        "context", ("dist", "inline")
    )
    def test_mask(self, benchmark, prefix, drop, shared_dist_ctx, lt_ctx, io_backend, context):
        io_backend = backends_by_name[io_backend]
        mib_hdr = os.path.join(prefix, MIB_FILE)
        flist = filelist(mib_hdr)

        if context == 'dist':
            ctx = shared_dist_ctx
        elif context == 'inline':
            ctx = lt_ctx
        else:
            raise ValueError

        ds = ctx.load(filetype="mib", path=mib_hdr, io_backend=io_backend)

        sig_shape = ds.shape.sig

        def mask():
            return np.ones(sig_shape, dtype=bool)

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
            iterations=1
        )

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
        "sparsity", (10, 16, 64, 128)
    )
    @pytest.mark.parametrize(
        "io_backend", ("mmap", "mmap_readahead", "buffered"),
    )
    def test_sparse_roi(self, benchmark, prefix, drop, io_backend, shared_dist_ctx, sparsity):
        io_backend = backends_by_name[io_backend]
        mib_hdr = os.path.join(prefix, MIB_FILE)
        flist = filelist(mib_hdr)

        ctx = shared_dist_ctx
        ds = ctx.load(filetype="mib", path=mib_hdr, io_backend=io_backend)

        sparse_roi = np.zeros(ds.shape.nav.size, dtype=bool)
        sparse_roi[::sparsity] = True

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
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds, roi=sparse_roi),
            warmup_rounds=0,
            rounds=1,
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
    mib_hdr = os.path.join(prefix, MIB_FILE)
    flist = filelist(mib_hdr)

    with api.Context() as ctx:
        ds = ctx.load(filetype="mib", path=mib_hdr, io_backend=io_backend)

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
