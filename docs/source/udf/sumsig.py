import numpy as np
from libertem.api import Context
from libertem.udf import UDF


class SumOverSig(UDF):
    def get_result_buffers(self, meta):
        """
        Describe the buffers we need to store our results:
        kind="nav" means we want to have a value for each coordinate
        in the navigation dimensions. We name our buffer 'pixelsum'.
        """
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        """
        Sum up all pixels in this frame and store the result in the pixelsum
        buffer. `self.results.pixelsum` is a view into the result buffer we
        defined above, and corresponds to the entry for the current frame we
        work on. We don't have to take care of finding the correct index for
        the frame we are processing ourselves.
        """
        self.results.pixelsum[:] = np.sum(frame)


# run the UDF on a dataset:
ctx = Context()
dataset = ctx.load("...")

res = ctx.run_udf(
   udf=SumOverSig(),
   dataset=dataset,
)
