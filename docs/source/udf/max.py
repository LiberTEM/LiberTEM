import numpy as np
from libertem.api import Context
from libertem.udf import UDF


class MaxUDF(UDF):
    def get_result_buffers(self):
        """
        Describe the buffers we need to store our results:
        kind="sig" means we want to have a value for each coordinate
        in the signal dimensions (i.e. a value for each pixel of the diffraction patterns).
        We name our buffer 'maxbuf'.
        """
        return {
            'maxbuf': self.buffer(
                kind="sig", dtype=self.meta.dataset_dtype
            )
        }

    def process_frame(self, frame):
        """
        In this function, we have a frame and the buffer `maxbuf` available, which we declared
        above. This function is called for all frames / diffraction patterns in the data set.
        The maxbuf is a partial result, and all partial results will later be merged (see below).

        In this case, we determine the maximum from the current maximum and the current frame, for
        each pixel in the diffraction pattern.

        Notes:

        - You cannot rely on any particular order of frames this function is called in.
        - Your function should be pure, that is, it should not have side effects and should
        only depend on it's input parameters.
        """
        self.results.maxbuf[:] = np.maximum(frame, self.results.maxbuf)

    def merge(self, dest, src):
        """
        merge two partial results, from src into dest
        """
        dest['maxbuf'][:] = np.maximum(dest['maxbuf'], src['maxbuf'])


ctx = Context()
ds = ctx.load("...")
res = ctx.run_udf(
    dataset=ds,
    udf=MaxUDF(),
)
