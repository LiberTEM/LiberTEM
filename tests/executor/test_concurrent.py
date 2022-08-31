import pytest
import numpy as np

from libertem.udf.stddev import StdDevUDF
from libertem.udf.masks import ApplyMasksUDF


@pytest.mark.parametrize(
    'use_roi', [True, False]
)
def test_concurrent_executor(lt_ctx, concurrent_ctx, default_raw, use_roi):
    if use_roi:
        roi = np.random.choice([True, False], default_raw.shape.nav)
    else:
        roi = None
    mask = np.random.random(default_raw.shape.sig)

    def mask_factory():
        return mask

    load_params = {
        'filetype': 'raw',
        'path': default_raw._path,
        'nav_shape': default_raw.shape.nav,
        'sig_shape': default_raw.shape.sig,
        'dtype': default_raw.dtype
    }

    udfs = [StdDevUDF(), ApplyMasksUDF(mask_factories=[mask_factory])]
    ref_res = lt_ctx.run_udf(dataset=default_raw, udf=udfs, roi=roi)
    ds = concurrent_ctx.load(**load_params)
    res = concurrent_ctx.run_udf(dataset=ds, udf=udfs, roi=roi)

    assert len(ref_res) == len(res)

    for index, value in enumerate(ref_res):
        for key, ref in value.items():
            print("index", index, "key", key)
            assert np.allclose(ref.data, res[index][key].data, equal_nan=True)
        for key in res[index].keys():
            assert key in value
