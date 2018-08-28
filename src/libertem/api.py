import psutil

from libertem.io.dataset import load
from libertem.job.masks import ApplyMasksJob
from libertem.executor.dask import DaskJobExecutor
from libertem.viz import visualize_simple


class Context:
    def __init__(self):
        """
        Create a new context. In the background, this creates a suitable
        executor and spins up a local Dask cluster.
        """
        self.executor = self._create_local_executor()

    def load(self, filetype, *args, **kwargs):
        """
        Load a `DataSet`. As it doesn't load the whole data into RAM at once,
        you can load and process datasets that are bigger than your available RAM.
        Using fast storage (i.e. SSD) is advisable.

        Parameters
        ----------
        filetype : str
            see libertem.io.dataset.filetypes for supported types, example: 'hdf5'


        Note
        ----

        Additional parameters are passed to the concrete DataSet implementation

        Returns
        -------
        DataSet
            a subclass of DataSet
        """
        # delegate to libertem.io.dataset.load:
        ds = load(filetype, *args, **kwargs)
        ds.check_valid()
        return ds

    def create_mask_job(self, factories, dataset):
        """
        Create a mask application job. Each factory function should, when called,
        return a numpy array with the same shape as frames in the dataset (so dataset.shape[2:]).

        Parameters
        ----------
        factories : list of mask factory functions
            functions that take no arguments and create masks
        dataset : DataSet
            dataset to work on

        Examples
        --------
        >>> job = ctx.create_mask_job(
        ... factories=[lambda: np.ones(dataset.shape[2:])],
        ... dataset=dataset)
        >>> result = ctx.run(job)
        """
        return ApplyMasksJob(
            dataset=dataset,
            mask_factories=factories,
        )

    def run(self, job, out=None):
        """
        Run the given `Job` and return the result data.

        Parameters
        ----------
        job : Job
            the job to run
        out : ndarray
            ndarray to store the result, if None it is created for you
        """
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
        """
        Normalize and apply a colormap to result_frames and return resulting RGB data
        """
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
