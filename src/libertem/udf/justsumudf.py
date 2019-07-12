import numpy as np

from libertem.udf import UDF


class justsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind="nav", dtype="float32"
            ),
        }

    def process_frame(self, frame):
        self.results.intensity[:] = np.sum(frame)


def run_justsum(ctx, dataset):
    udf = justsumUDF()
    pass_results = ctx.run_udf(dataset=dataset, udf=udf)
    return pass_results
