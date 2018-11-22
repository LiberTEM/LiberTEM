import psutil
from typing import Union

import numpy as np
from libertem.io.dataset import load, filetypes
from libertem.io.dataset.base import DataSet
from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.job.base import Job
from libertem.common.slice import Slice
from libertem.executor.dask import DaskJobExecutor
from libertem.analysis.com import COMAnalysis
from libertem.analysis.disk import DiskMaskAnalysis
from libertem.analysis.ring import RingMaskAnalysis
from libertem.analysis.sum import SumAnalysis
from libertem.analysis.point import PointMaskAnalysis
from libertem.analysis.masks import MasksAnalysis
from libertem.analysis.base import BaseAnalysis


class Context:
    """
    Context is the main entry point of the LiberTEM API. It contains
    methods for loading datasets, creating jobs on them and running
    them.
    """

    def __init__(self, executor=None):
        """
        Create a new context. In the background, this creates a suitable
        executor and spins up a local Dask cluster.
        """
        if executor is None:
            executor = self._create_local_executor()
        self.executor = executor

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
        Create a low-level mask application job. Each factory function should, when called,
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

    def create_mask_analysis(self, factories, dataset):
        """
        Create a mask application analysis. Each factory function should, when called,
        return a numpy array with the same shape as frames in the dataset (so dataset.shape[2:]).

        This is a more high-level method than `create_mask_job` and differs in the way the result
        is returned. With `create_mask_job`, it is a single numpy array, here we split it up for
        each mask we apply, make some default visualization available etc.

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
        >>> job = ctx.create_mask_analysis(
        ... factories=[lambda: np.ones(dataset.shape[2:])],
        ... dataset=dataset)
        >>> result = ctx.run(job)
        >>> result.mask_0.raw_data
        """
        return MasksAnalysis(
            dataset=dataset,
            parameters={
                'factories': factories,
            }
        )

    def create_com_analysis(self, dataset, cx: int, cy: int, mask_radius: int = None):
        """
        Perform a center-of-mass (first moment) analysis, possibly masked.

        Parameters
        ----------
        dataset
            the dataset to work on
        cx
            reference center x value
        cy
            reference center y value
        mask_radius
            mask out intensity outside of mask_radius from (cy, cx)
        """
        if mask_radius is None:
            mask_radius = float('inf')
        analysis = COMAnalysis(
            dataset=dataset,
            parameters={
                'cx': cx,
                'cy': cy,
                'r': mask_radius,
            },
        )
        return analysis

    def create_disk_analysis(self, dataset, cx: int, cy: int, r: int):
        """
        Integrate over a disk (i.e. filled circle)

        Parameters
        ----------
        dataset
            the dataset to work on
        cx
            center x value
        cy
            center y value
        r
            radius of the disk
        """
        return DiskMaskAnalysis(dataset=dataset, parameters={
            'cx': cx,
            'cy': cy,
            'r': r,
        })

    def create_ring_analysis(self, dataset, cx: int, cy: int, ri: int, ro: int):
        """
        Integrate over a ring

        Parameters
        ----------
        dataset
            the dataset to work on
        cx
            center x value
        cy
            center y value
        ri
            inner radius
        ro
            outer radius
        """
        return RingMaskAnalysis(dataset=dataset, parameters={
            'cx': cx,
            'cy': cy,
            'ri': ri,
            'ro': ro,
        })

    def create_point_analysis(self, dataset, x: int, y: int):
        """
        Select the pixel with coords (y, x) from each frame
        """
        return PointMaskAnalysis(dataset=dataset, parameters={
            'cx': x,
            'cy': y,
        })

    def create_sum_analysis(self, dataset):
        """
        Sum over all frames

        Parameters
        ----------
        dataset
            the dataset to work on
        """
        return SumAnalysis(dataset=dataset, parameters={})

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

    def run(self, job: Union[Job, BaseAnalysis]):
        """
        Run the given `Job` or `Analysis` and return the result data.

        Parameters
        ----------
        job
            the job or analysis to run
        """
        analysis = None
        if hasattr(job, 'get_job'):
            analysis = job
            job_to_run = analysis.get_job()
        else:
            job_to_run = job

        out = job_to_run.get_result_buffer()
        for tiles in self.executor.run_job(job_to_run):
            for tile in tiles:
                tile.copy_to_result(out)
        if analysis is not None:
            return analysis.get_results(out)
        return out

    def _create_local_executor(self):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = 2
        return DaskJobExecutor.make_local(cluster_kwargs={
            "threads_per_worker": 1,
            "n_workers": cores,
        })
