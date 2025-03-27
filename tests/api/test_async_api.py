import numpy as np
import pytest

from libertem.udf.sum import SumUDF
from libertem.api import Context


@pytest.mark.asyncio
async def test_run_udf_async(default_raw, lt_ctx: Context):
    async with lt_ctx.run_udf_async(dataset=default_raw, udf=SumUDF()) as run:
        async for update in run.updates():
            latest_result = await run.get_latest()

    # here, `latest_result` should equal the "final" result:
    res_reference = lt_ctx.run_udf(dataset=default_raw, udf=SumUDF())

    assert np.allclose(
        res_reference['intensity'].data,
        latest_result.buffers[0]['intensity'].data
    )


# TODO:
# - [ ] Add a test with `update_parameters_experimental`
# - [ ] Add a test that ensures lazyness (if we only access `.buffers` once,
#       `get_results()` should only be called once, too!)
# - [ ] any more tests? can we ensure we are copying in `get_latest`?
