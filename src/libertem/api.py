import psutil

import numpy as np
from libertem.io.dataset import load, filetypes
from libertem.io.dataset.base import DataSet
from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.job.base import Job
from libertem.common.slice import Slice
from libertem.executor.dask import DaskJobExecutor


class Context:
    """
    Context is the main entry point of the LiberTEM API. It contains
    methods for loading datasets, creating jobs on them and running
    them.
    """

    def __init__(self):
        """
        Create a new context. In the background, this creates a suitable
        executor and spins up a local Dask cluster.
        """
        self.executor = self._create_local_executor()

    def load(self, filetype: str, *args, **kwargs) -> DataSet:
        """
        Load a `DataSet`. As it doesn't load the whole data into RAM at once,
        you can load and process datasets that are bigger than your available RAM.
        Using fast storage (i.e. SSD) is advisable.

        Parameters
        ----------
        filetype : str
            one of: %(types)s
        args
            passed on to the DataSet implementation
        kwargs
            passed on to the DataSet implementation

        Returns
        -------
        DataSet
            the loaded dataset

        Note
        ----

        Additional parameters are passed to the concrete DataSet implementation
        """
        # delegate to libertem.io.dataset.load:
        ds = load(filetype, *args, **kwargs)
        ds.check_valid()
        return ds

    load.__doc__ = load.__doc__ % {'types': ", ".join(filetypes.keys())}

    def create_mask_job(self, factories, dataset):
        """
        Create a mask application job. Each factory function should, when called,
        return a numpy array with the same shape as frames in the dataset (so dataset.shape[2:]).

        Parameters
        ----------
        factories
            list of functions that take no arguments and create masks
        dataset
            dataset to work on

        Examples
        --------
        >>> from libertem.api import Context
        >>> ctx = Context()
        >>> ds = ctx.load("...")
        >>> job = ctx.create_mask_job(
        ... factories=[lambda: np.ones(dataset.shape[2:])],
        ... dataset=dataset)
        >>> result = ctx.run(job)
        """
        return ApplyMasksJob(
            dataset=dataset,
            mask_factories=factories,
        )

    def create_pick_job(self, dataset, y: int, x: int) -> np.ndarray:
        """
        Pick a full frame at scan coordinates (y, x)

        Parameters
        ----------
        dataset
            the dataset to work on
        y
            the y coordinate of the frame
        x
            the x coordinate of the frame

        Returns
        -------
        :py:class:`numpy.ndarray`
            the frame as numpy array
        """
        shape = dataset.shape
        return PickFrameJob(
            dataset=dataset,
            slice_=Slice(
                origin=(y, x, 0, 0),
                shape=(1, 1) + shape[2:],
            ),
            squeeze=True,
        )

    def run(self, job: Job, out: np.ndarray = None):
        """
        Run the given `Job` and return the result data.

        Parameters
        ----------
        job
            the job to run
        out : :py:class:`numpy.ndarray`
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

    def _create_local_executor(self):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = 2
        return DaskJobExecutor.make_local(cluster_kwargs={
            "threads_per_worker": 1,
            "n_workers": cores,
        })
