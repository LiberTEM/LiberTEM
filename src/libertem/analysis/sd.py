import numpy as np

from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from libertem.udf import UDF
from libertem import masks
import libertem.udf.stddev as std

class SDAnalysis(BaseAnalysis):
    TYPE = 'UDF'

    def get_udf(self):
        return std.StdDevUDF()

    def get_roi(self):
        if "roi" not in self.parameters:
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
        if params["shape"] == "rect":
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
        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results.var,
                           visualized=visualize_simple(
                               udf_results.var, logarithmic=True),
                           key="intensity", title="intensity",
                           desc="SD"),
        ])