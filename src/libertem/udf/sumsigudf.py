import numpy as np

from libertem.udf import UDF


class SumSigUDF(UDF):
    """
    Sum over the signal axes. For each navigation position, the sum of all pixels is calculated.

    Examples
    --------
    >>> udf = SumSigUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> np.array(result["intensity"]).shape
    (16, 16)
    """

    def get_backends(self):
        return self.BACKEND_ALL

    def get_result_buffers(self):
        ""
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            'intensity': self.buffer(
                kind="nav", dtype=dtype, where='device'
            ),
        }

    def process_tile(self, tile):
        ""
        self.results.intensity[:] += self.forbuf(
            np.sum(
                # Flatten and sum axis 1 for cupyx.scipy.sparse support
                tile.reshape((tile.shape[0], -1)),
                axis=1
            ),
            self.results.intensity
        )


def run_sumsig(ctx, dataset):
    udf = SumSigUDF()
    pass_results = ctx.run_udf(dataset=dataset, udf=udf)
    return pass_results
