import numpy as np

from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from libertem.udf import UDF
from libertem import masks


class SumUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(kind='sig', dtype=self.params.dtype)
        }

    def process_tile(self, tile, tile_slice):
        self.results.intensity[:] += np.sum(tile, axis=0)

    def merge(self, dest, src):
        dest['intensity'][:] += src['intensity']


class SumAnalysis(BaseAnalysis):
    TYPE = 'UDF'

    def get_udf(self):
        dest_dtype = np.dtype(self.dataset.dtype)
        if dest_dtype.kind not in ('c', 'f'):
            dest_dtype = 'float32'
        return SumUDF(dtype=dest_dtype)

    def get_roi(self):
        if "shape" not in self.parameters["roi"]:
            return None
        params = self.parameters["roi"]
        ny, nx = tuple(self.dataset.shape.nav)
        if params["shape"] == "disk":
            roi = masks.circular(
                params["cx"],
                params["cy"],
                nx, ny,
                params["r"],
            )
        elif params["shape"] == "rect":
            roi = masks.rectangular(
                params["x"],
                params["y"],
                params["width"],
                params["height"],
                nx, ny,
            )
        else:
            raise NotImplementedError("unknown shape %s" % params["shape"])
        return roi

    def get_udf_results(self, udf_results, roi):
        if udf_results.intensity.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    udf_results.intensity,
                    key_prefix="intensity",
                    title="intensity",
                    desc="sum of all frames",
                )
            )

        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results.intensity,
                           visualized=visualize_simple(udf_results.intensity, logarithmic=True),
                           key="intensity", title="intensity", desc="sum of frames"),
        ])
