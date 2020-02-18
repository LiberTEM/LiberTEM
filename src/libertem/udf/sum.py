import numpy as np

from libertem.udf import UDF


class SumUDF(UDF):
    """
    Sum up frames, preserving the signal dimension

    Examples
    --------
    >>> udf = SumUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> np.array(result["intensity"]).shape
    (16, 16)
    """
    def __init__(self, dtype='float32'):
        ''
        super().__init__(dtype=dtype)

    def get_result_buffers(self):
        ''
        return {
            'intensity': self.buffer(kind='sig', dtype=self.params.dtype)
        }

    def process_tile(self, tile):
        ''
        self.results.intensity[:] += np.sum(tile, axis=0)

    def merge(self, dest, src):
        ''
        dest['intensity'][:] += src['intensity']