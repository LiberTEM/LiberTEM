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

    def get_result_buffers(self):
        ""
        return {
            'intensity': self.buffer(
                kind="nav", dtype="float32"
            ),
        }

    def process_frame(self, frame):
        ""
        self.results.intensity[:] = np.sum(frame)


def run_sumsig(ctx, dataset):
    udf = SumSigUDF()
    pass_results = ctx.run_udf(dataset=dataset, udf=udf)
    return pass_results
