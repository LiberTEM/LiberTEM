from typing import Union, Tuple

import psutil
import numpy as np
from libertem.io.dataset import load, filetypes
from libertem.io.dataset.base import DataSet
from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.job.base import Job
from libertem.common import Slice, Shape
from libertem.executor.dask import DaskJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
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
        ds = self.executor.run_function(load, filetype, *args, **kwargs)
        ds = self.executor.run_function(ds.initialize)
        self.executor.run_function(ds.check_valid)
        return ds

    load.__doc__ = load.__doc__ % {"types": ", ".join(filetypes.keys())}

    def create_mask_job(self, factories, dataset, use_sparse=None):
        """
        Create a low-level mask application job. Each factory function should, when called,
        return a numpy array with the same shape as frames in the dataset (so dataset.shape.sig).

        Parameters
        ----------
        factories
            list of functions that take no arguments and create masks. The returned masks can be
            numpy arrays or scipy.sparse matrices. The mask factories should not reference large
            objects because they can create significant overheads when they are pickled and
            unpickled.
        dataset
            dataset to work on
        use_sparse
            * None (default): Use sparse matrix multiplication if all factory functions return a \
            sparse mask, otherwise convert all masks to dense matrices and use dense matrix \
            multiplication
            * True: Convert all masks to sparse matrices.
            * False: Convert all masks to dense matrices.

        Examples
        --------
        >>> from libertem.api import Context
        >>> ctx = Context()
        >>> ds = ctx.load("...")
        >>> # Use intermediate variables instead of referencing
        >>> # large complex objects like a dataset within the
        >>> # factory function
        >>> shape = dataset.shape.sig
        >>> job = ctx.create_mask_job(
        ... factories=[lambda: np.ones(shape)],
        ... dataset=dataset)
        >>> result = ctx.run(job)
        """
        return ApplyMasksJob(
            dataset=dataset, mask_factories=factories, use_sparse=use_sparse
        )

    def create_mask_analysis(self, factories, dataset, use_sparse=None):
        """
        Create a mask application analysis. Each factory function should, when called,
        return a numpy array with the same shape as frames in the dataset (so dataset.shape.sig).

        This is a more high-level method than `create_mask_job` and differs in the way the result
        is returned. With `create_mask_job`, it is a single numpy array, here we split it up for
        each mask we apply, make some default visualization available etc.

        Parameters
        ----------
        factories
            list of functions that take no arguments and create masks. The returned masks can be
            numpy arrays or scipy.sparse matrices. The mask factories should not reference large
            objects because they can create significant overheads when they are pickled and
            unpickled.
        dataset
            dataset to work on
        use_sparse
            * None (default): Use sparse matrix multiplication if all factory functions return a \
            sparse mask, otherwise convert all masks to dense matrices and use dense matrix \
            multiplication
            * True: Convert all masks to sparse matrices.
            * False: Convert all masks to dense matrices.

        Examples
        --------
        >>> from libertem.api import Context
        >>> ctx = Context()
        >>> ds = ctx.load("...")
        >>> # Use intermediate variables instead of referencing
        >>> # large complex objects like a dataset within the
        >>> # factory function
        >>> shape = dataset.shape.sig
        >>> job = ctx.create_mask_analysis(
        ... factories=[lambda: np.ones(shape)],
        ... dataset=dataset)
        >>> result = ctx.run(job)
        >>> result.mask_0.raw_data
        """
        return MasksAnalysis(
            dataset=dataset,
            parameters={"factories": factories, "use_sparse": use_sparse},
        )

    def create_com_analysis(self, dataset, cx: int = None, cy: int = None, mask_radius: int = None):
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
        if dataset.shape.nav.dims != 2:
            raise ValueError("incompatible dataset: need two navigation dimensions")
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {name: loc[name] for name in ['cx', 'cy'] if loc[name] is not None}
        if mask_radius is not None:
            parameters['r'] = mask_radius
        analysis = COMAnalysis(
            dataset=dataset, parameters=parameters
        )
        return analysis

    def create_disk_analysis(self, dataset, cx: int = None, cy: int = None, r: int = None):
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
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {name: loc[name] for name in ['cx', 'cy', 'r'] if loc[name] is not None}
        return DiskMaskAnalysis(
            dataset=dataset, parameters=parameters
        )

    def create_ring_analysis(
            self, dataset, cx: int = None, cy: int = None, ri: int = None, ro: int = None):
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
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {name: loc[name] for name in ['cx', 'cy', 'ri', 'ro'] if loc[name] is not None}
        return RingMaskAnalysis(
            dataset=dataset, parameters=parameters
        )

    def create_point_analysis(self, dataset, x: int = None, y: int = None):
        """
        Select the pixel with coords (y, x) from each frame
        """
        if dataset.shape.nav.dims != 2:
            raise ValueError("incompatible dataset: need two navigation dimensions")
        parameters = {
            'cx': x,
            'cy': y,
        }
        parameters = {
            k: v
            for k, v in parameters.items()
            if v is not None
        }
        return PointMaskAnalysis(dataset=dataset, parameters=parameters)

    def create_sum_analysis(self, dataset):
        """
        Sum of all signal elements

        Parameters
        ----------
        dataset
            the dataset to work on
        """
        return SumAnalysis(dataset=dataset, parameters={})

    def create_pick_job(self, dataset, origin: Tuple[int], shape: Tuple[int] = None) -> np.ndarray:
        """
        Pick raw data from `origin` with the size defined in `shape`.

        NOTE: if you just want to read single frames, it is easier to use `create_pick_analysis`.

        NOTE: It is not efficient to use this method on large parts of datasets, please consider
        implementing an analysis instead.

        Parameters
        ----------
        dataset
            The dataset to work on
        origin
            Where to start reading. You can either specify all dimensions, or only nav dimensions,
            in which case the signal is read starting from (0, ..., 0).
        shape
            The shape of the data to read. If None, read a "frame" or single signal element

        Returns
        -------
        :py:class:`numpy.ndarray`
            the raw data as numpy array

        Examples
        --------
        >>> from libertem.api import Context
        >>> ctx = Context()
        >>> ds = ctx.load("...")
        >>> origin = (7, 8, 9)
        >>> job = create_pick_job(dataset=ds, origin=origin)
        >>> result = ctx.run(job)
        >>> assert result.shape == ds.shape.sig

        """
        # FIXME: this method works well if we can flatten to 3D
        # need vectorized I/O for general case
        raw_shape = dataset.raw_shape
        if len(origin) == dataset.shape.nav.dims:
            if raw_shape.dims != len(origin) and raw_shape.nav.dims == 1:
                origin = (np.ravel_multi_index(origin, dataset.shape.nav),)
            origin = origin + tuple([0] * dataset.shape.sig.dims)
        elif len(origin) == dataset.shape.dims:
            if raw_shape.dims != len(origin) and raw_shape.nav.dims == 1:
                origin = (np.ravel_multi_index(origin, dataset.shape),)
        else:
            raise ValueError(
                "incompatible dataset: origin needs to match dataset shape (%r)" % dataset.shape
            )
        if shape is None:
            shape = tuple([1] * raw_shape.nav.dims) + tuple(raw_shape.sig)
        else:
            if len(shape) != dataset.shape.dims:
                raise ValueError(
                    "incompatible: shape needs to match the dataset shape"
                )
        shape = Shape(shape, sig_dims=raw_shape.sig.dims)
        if raw_shape.dims != len(shape) and raw_shape.nav.dims == 1:
            shape = shape.flatten_nav()
        slice_ = Slice(origin=origin,
                       shape=Shape(shape, sig_dims=raw_shape.sig.dims))
        return PickFrameJob(
            dataset=dataset,
            slice_=slice_,
            squeeze=True,
        )

    def create_pick_analysis(self, dataset, x: int, y: int = None, z: int = None):
        """
        Pick a single frame / signal element from (z, y, x). Number of parameters
        must match number of navigation dimensions in the dataset, for example if
        you have a 4D dataset with two signal dimensions and two navigation dimensions,
        you need to specify x and y.

        Parameters
        ----------
        dataset
            The dataset to work on
        x
            x coordinate
        y
            y coordinate
        z
            z coordinate

        Returns
        -------
        :py:class:`numpy.ndarray`
            the frame as numpy array

        Examples
        --------
        >>> from libertem.api import Context
        >>> ctx = Context()
        >>> ds = ctx.load("...")
        >>> origin = (7, 8, 9)
        >>> job = create_pick_analysis(dataset=ds, x=9, y=8, z=7)
        >>> result = ctx.run(job)
        >>> assert result.intensity.raw_data.shape == ds.shape.sig
        """
        loc = locals()
        parameters = {name: loc[name] for name in ['x', 'y', 'z'] if loc[name] is not None}
        return PickFrameAnalysis(dataset=dataset, parameters=parameters)

    def run(self, job: Union[Job, BaseAnalysis]):
        """
        Run the given `Job` or `Analysis` and return the result data.

        Parameters
        ----------
        job
            the job or analysis to run
        """
        analysis = None
        if hasattr(job, "get_job"):
            analysis = job
            job_to_run = analysis.get_job()
        else:
            job_to_run = job

        out = job_to_run.get_result_buffer()
        for tiles in self.executor.run_job(job_to_run):
            for tile in tiles:
                tile.reduce_into_result(out)
        if analysis is not None:
            return analysis.get_results(out)
        return out

    def _create_local_executor(self):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = 2
        return DaskJobExecutor.make_local(
            cluster_kwargs={"threads_per_worker": 1, "n_workers": cores}
        )

    def close(self):
        self.executor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
