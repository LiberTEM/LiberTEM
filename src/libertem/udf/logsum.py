import numpy as np

from libertem.udf import UDF


class LogsumUDF(UDF):
    """
    Sum up logscaled frames

    In comparison to log-scaling the sum, this highlights regions with slightly higher
    intensity that appear in many frames in relation to very high intensity in a few frames.

    Examples
    --------
    >>> udf = LogsumUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> np.array(result["logsum"]).shape
    (32, 32)
    """
    def __init__(self):
        super().__init__()

    def get_backends(self):
        return [
            b for b in self.BACKEND_ALL
            if b not in (
                self.BACKEND_SCIPY_COO, self.BACKEND_SCIPY_CSR, self.BACKEND_SCIPY_CSC,
                self.BACKEND_SCIPY_COO_ARRAY,
                self.BACKEND_SCIPY_CSR_ARRAY,
                self.BACKEND_SCIPY_CSC_ARRAY,
                self.BACKEND_CUPY_SCIPY_COO, self.BACKEND_CUPY_SCIPY_CSC,
                self.BACKEND_CUPY_SCIPY_CSR,
            )
        ]

    def get_result_buffers(self):
        ""
        return {
            'logsum': self.buffer(
                kind='sig', dtype='float32', where='device'
            ),
        }

    def merge(self, dest, src):
        ""
        dest.logsum[:] += src.logsum[:]

    def merge_all(self, ordered_results):
        ''
        chunks = [b.logsum for b in ordered_results.values()]
        logsum = np.stack(chunks, axis=0).sum(axis=0)
        return {'logsum': logsum}

    def process_frame(self, frame):
        ""
        self.results.logsum[:] += self.forbuf(
            np.log(frame - np.min(frame) + 1),
            self.results.logsum
        )


def run_logsum(ctx, dataset, roi=None):
    '''
    Sum up logscaled frames

    In comparison to log-scaling the sum, this highlights regions with slightly higher
    intensity that appear in many frames in relation to very high intensity in a few frames.

    Example:

    f1 = (11, 101)
    f2 = (11, 1)
    f2 = (11, 1)
    ...
    f10 = (11, 1)

    log10(sum(f1 ... f10)) == (2.04, 2.04)

    sum(log10(f1) ... log10(f10)) == (10.4, 2.04)

    '''
    udf = LogsumUDF()
    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
