import pytest
import numpy as np

from libertem.udf.masks import ApplyMasksUDF


@pytest.mark.benchmark(
    group="udf"
)
def test_masks_udf(large_raw, shared_dist_ctx, benchmark):
    sig_shape = tuple(large_raw.shape.sig)

    def mask_factory():
        return np.ones(sig_shape)
    udf = ApplyMasksUDF(mask_factories=[mask_factory], mask_count=1, mask_dtype=np.float64)

    benchmark(shared_dist_ctx.run_udf, udf=udf, dataset=large_raw)
