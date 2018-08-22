import psutil

from libertem.io.dataset import load
from libertem.job.masks import ApplyMasksJob
from libertem.executor.dask import DaskJobExecutor
from libertem.viz import visualize_simple


class Context:
    def __init__(self):
        self.executor = self._create_local_executor()

    def load(self, *args, **kwargs):
        # delegate to libertem.io.dataset.load:
        return load(*args, **kwargs)

    def create_mask_job(self, factories, dataset):
        return ApplyMasksJob(
            dataset=dataset,
            mask_factories=factories,
        )

    def run(self, job, out=None):
        if out is None:
            out = job.get_result_buffer()
        else:
            # TODO: assert out.shape == job.get_result_shape()
            # and/or try to reshape out into the right shape
            pass
        for tiles in self.executor.run_job(job):
            for tile in tiles:
                tile.copy_to_result(out)
        return out

    def apply_colormap(self, result_frames):
        return [visualize_simple(result_frame)
                for result_frame in result_frames]

    def _create_local_executor(self):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = 2
        return DaskJobExecutor.make_local(cluster_kwargs={
            "threads_per_worker": 1,
            "n_workers": cores,
        })
