import os

import pytest
import numpy as np

from libertem import api
from libertem.udf.masks import ApplyMasksUDF

from cache_utils import drop_cache, warmup_cache


NET_PREFIX = "/storage/holo/clausen/testdata/ER-C-1/groups/data_science/data/reference/"
SSD_PREFIX = "/cachedata/users/libertem/reference/"
HDD_PREFIX = "/data/users/libertem/reference/"

MIB_FILE = "MIB/20200518 165148/default.hdr"


def filelist(mib_hdr):
    mib_dir = os.path.dirname(mib_hdr)
    return [os.path.join(mib_dir, fname) for fname in os.listdir(mib_dir)]


@pytest.mark.parametrize(
    "drop", ("cold_cache", "warm_cache")
)
@pytest.mark.parametrize(
    "prefix", (SSD_PREFIX, HDD_PREFIX, NET_PREFIX)
)
def test_sequential(benchmark, prefix, drop, lt_ctx):
    mib_hdr = prefix + MIB_FILE

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


@pytest.mark.parametrize(
    "first", ("cold_executor", "warm_executor")
)
@pytest.mark.parametrize(
    "drop", ("cold_cache", "warm_cache")
)
@pytest.mark.parametrize(
    "prefix", (SSD_PREFIX,  HDD_PREFIX, NET_PREFIX)
)
def test_mask(benchmark, prefix, drop, first):
    mib_hdr = prefix + MIB_FILE
    flist = filelist(mib_hdr)

    # we always start with a fresh context
    # to also capture
    with api.Context() as ctx:
        ds = ctx.load(filetype="auto", path=mib_hdr)

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask])

        if first == "warm_executor":
            ctx.run_udf(udf=udf, dataset=ds)
        elif first == "cold_executor":
            pass
        else:
            raise ValueError("bad param")

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
