import os

import pytest
import numpy as np

from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.stddev import StdDevUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF

from utils import get_testdata_prefixes

RAW_CSR_FILE = "raw_csr/sparse.toml"

PREFIXES = get_testdata_prefixes()


class TestUseSharedExecutor:
    @pytest.mark.benchmark(
        group="io"
    )
    @pytest.mark.parametrize(
        "prefix", PREFIXES
    )
    @pytest.mark.parametrize(
        "context", ("inline", "dist")
    )
    @pytest.mark.parametrize(
        "udf_kind", ("stddev", )
    )
    def test_mask(self, benchmark, prefix, shared_dist_ctx, lt_ctx, context, udf_kind):
        print("prefix", prefix)
        print("context", context)
        print("udf_kind", udf_kind)
        sparse_toml = os.path.join(prefix, RAW_CSR_FILE)

        if context == 'dist':
            ctx = shared_dist_ctx
        elif context == 'inline':
            ctx = lt_ctx
        else:
            raise ValueError

        ds = ctx.load(filetype="raw_csr", path=sparse_toml)

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        if udf_kind == 'mask':
            udf = ApplyMasksUDF(mask_factories=[mask])
        elif udf_kind == 'stddev':
            udf = StdDevUDF()
        elif udf_kind == 'sumsig':
            udf = SumSigUDF()
        elif udf_kind == 'sum':
            udf = SumUDF()
        else:
            raise ValueError('unknown param')

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=1,
            rounds=1,
            iterations=1
        )
