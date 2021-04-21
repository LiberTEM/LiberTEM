import numpy as np

from libertem.udf import UDF


class SumUDF(UDF):
    """
    Sum up frames, preserving the signal dimension

    Parameters
    ----------
    dtype : numpy.dtype, optional
        Preferred dtype for computation, default 'float32'. The actual dtype will be determined
        from this value and the dataset's dtype using :meth:`numpy.result_type`.
        See also :ref:`udf dtype`.

    Examples
    --------
    >>> udf = SumUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> np.array(result["intensity"]).shape
    (32, 32)
    """
    def __init__(self, dtype='float32'):
        super().__init__(dtype=dtype)

    def get_preferred_input_dtype(self):
        return self.params.dtype

    def get_result_buffers(self):
        ''
        return {
            'intensity': self.buffer(kind='sig', dtype=self.meta.input_dtype)
        }

    def process_tile(self, tile):
        ''
        self.results.intensity[:] += np.sum(tile, axis=0)

    def merge(self, dest, src):
        ''
        dest.intensity[:] += src.intensity
