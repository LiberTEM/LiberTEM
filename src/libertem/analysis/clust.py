from skimage.feature import peak_local_max
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import numpy as np
import sparse

from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from libertem.utils.async_utils import run_blocking
from libertem.executor.base import JobCancelledError
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.base import UDFRunner
from libertem.udf.stddev import StdDevUDF, consolidate_result
from libertem import masks
from libertem.masks import _make_circular_mask


class ClusterAnalysis(BaseAnalysis):
    TYPE = "UDF"

    def get_udf(self):
        # FIXME: we don't have all parameters available here to actually construct
        # a useful udf instance - this should be optional for Analysis subclasses
        # that implement the `controller` method
        return None

    def get_udf_results(self, udf_results, roi):
        n_clust = self.parameters["n_clust"]
        y, x = tuple(self.dataset.shape.nav)
        connectivity = grid_to_graph(
            # Transposed!
            n_x=y,
            n_y=x
        )
        f = udf_results['intensity'].raw_data
        feature_vector = f / np.abs(f).mean(axis=0)

        clustering = AgglomerativeClustering(
            affinity='euclidean', n_clusters=n_clust, linkage='ward',
            connectivity=connectivity
        ).fit(feature_vector)
        labels = np.array(clustering.labels_+1)
        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'].data,
                           visualized=visualize_simple(
                               labels.reshape(self.dataset.shape.nav)),
                           key="intensity", title="intensity",
                           desc="result from integration over mask in Fourier space"),
        ])

    def get_sd_roi(self):
        if "roi" not in self.parameters:
            return None
        params = self.parameters["roi"]
        ny, nx = tuple(self.dataset.shape.nav)
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

    async def controller(self, cancel_id, executor, job_is_cancelled, send_results):
        stddev_udf = StdDevUDF()

        roi = self.get_sd_roi()

        result_iter = UDFRunner(stddev_udf).run_for_dataset_async(
            self.dataset, executor, roi=roi, cancel_id=cancel_id
        )
        async for sd_udf_results in result_iter:
            pass

        if job_is_cancelled():
            raise JobCancelledError()

        sd_udf_results = consolidate_result(sd_udf_results)

        center = (self.parameters["cy"], self.parameters["cx"])
        rad_in = self.parameters["ri"]
        rad_out = self.parameters["ro"]
        n_peaks = self.parameters["n_peaks"]
        min_dist = self.parameters["min_dist"]
        sstd = sd_udf_results['std']
        sshape = sstd.shape
        if not (center is None or rad_in is None or rad_out is None):
            mask_out = 1*_make_circular_mask(center[1], center[0], sshape[1], sshape[0], rad_out)
            mask_in = 1*_make_circular_mask(center[1], center[0], sshape[1], sshape[0], rad_in)
            mask = mask_out - mask_in
            masked_sstd = sstd*mask
        else:
            masked_sstd = sstd

        coordinates = peak_local_max(masked_sstd, num_peaks=n_peaks, min_distance=min_dist)

        y = coordinates[..., 0]
        x = coordinates[..., 1]
        z = range(len(y))

        mask = sparse.COO(
            shape=(len(y), ) + tuple(self.dataset.shape.sig),
            coords=(z, y, x), data=1
        )

        udf = ApplyMasksUDF(
            mask_factories=lambda: mask, mask_count=len(y), mask_dtype=np.uint8,
            use_sparse=True
        )

        result_iter = UDFRunner(udf).run_for_dataset_async(
            self.dataset, executor, cancel_id=cancel_id
        )
        async for udf_results in result_iter:
            pass

        if job_is_cancelled():
            raise JobCancelledError()

        results = await run_blocking(
            self.get_udf_results,
            udf_results=udf_results,
            roi=roi,
        )
        await send_results(results, True)
