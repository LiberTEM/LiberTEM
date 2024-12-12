from typing import (
    TYPE_CHECKING, Any, Optional, Union, overload
)
from collections.abc import Iterable, Generator, Coroutine
from typing_extensions import Literal
import os
import pathlib
import warnings
import weakref
import atexit
import logging

from opentelemetry import trace
import numpy as np
from libertem.executor.pipelined import PipelinedExecutor

from libertem.io.corrections import CorrectionSet
from libertem.executor.concurrent import ConcurrentJobExecutor
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset import load, filetypes
from libertem.io.dataset.base import DataSet
from libertem.common.buffers import BufferWrapper
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from libertem.executor.delayed import DelayedJobExecutor
from libertem.executor.integration import get_dask_integration_executor
from libertem.common.executor import JobExecutor
from libertem.common.progress import ProgressReporter
from libertem.masks import MaskFactoriesType
from libertem.analysis.raw import PickFrameAnalysis
from libertem.analysis.com import COMAnalysis
from libertem.analysis.radialfourier import RadialFourierAnalysis
from libertem.analysis.disk import DiskMaskAnalysis
from libertem.analysis.ring import RingMaskAnalysis
from libertem.analysis.sum import SumAnalysis
from libertem.analysis.point import PointMaskAnalysis
from libertem.analysis.masks import MasksAnalysis
from libertem.analysis.base import AnalysisResultSet, Analysis
from libertem.udf.base import UDFResultDict, UDF, UDFResults, UDFRunner
from libertem.udf.auto import AutoUDF
from libertem.common.async_utils import async_generator, run_agen_get_last, run_gen_get_last
from libertem.common.sparse import sparse_to_coo, to_dense
from libertem.common.exceptions import ExecutorSpecException

if TYPE_CHECKING:
    import numpy.typing as nt
    from sparse import SparseArray
    from scipy.sparse import spmatrix
    from libertem.viz.base import Live2DPlot

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

RunUDFResultType = UDFResultDict
RunUDFSyncL = list[UDFResultDict]
RunUDFAsync = Coroutine[None, None, UDFResultDict]
RunUDFAsyncL = Coroutine[None, None, list[UDFResultDict]]
ExecutorSpecType = Literal[
    'synchronous',
    'inline',
    'threads',
    'dask',
    'dask-integration',
    'dask-make-default',
    'delayed',
    'pipelined',
]
IterableRoiT = Iterable[tuple[tuple[int, ...], bool]]
RoiT = Optional[Union[np.ndarray, 'SparseArray', 'spmatrix', tuple[int, ...], IterableRoiT]]


class ResultGenerator:
    """
    Yields partial results from one or more UDFs currently running,
    with methods to control some aspects of the UDF run.
    """

    def __init__(
        self,
        task_results: Generator[UDFResults, None, None],
        runner: UDFRunner,
        result_iter,
    ):
        self._task_results = task_results
        self._runner = runner
        self._result_iter = result_iter

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._task_results)

    def close(self):
        self._task_results.close()
        self._result_iter.close()

    def update_parameters_experimental(self, parameters: list[dict[str, Any]]):
        """
        Update parameters while the UDFs are running.

        :code:`parameters` should be a list of dicts, with one dict for each
        UDF you are running.

        The dicts should only contain items for those parameters you want to
        update.
        """
        logger.debug("ResultGenerator.update_parameters_experimental: %s", parameters)
        self._result_iter.update_parameters_experimental(parameters)

    def throw(self, exc: Exception):
        return self._result_iter.throw(exc)


class ResultAsyncGenerator:
    """
    async wrapper of `ResultGenerator`.
    """

    def __init__(self, result_generator: ResultGenerator):
        from concurrent.futures import ThreadPoolExecutor
        self._result_generator = result_generator
        self._pool = ThreadPoolExecutor(max_workers=1)

    def __aiter__(self):
        return self

    async def __anext__(self):
        from .common.async_utils import MyStopIteration
        try:
            return await self._run_in_executor(self._inner_next)
        except MyStopIteration:
            raise StopAsyncIteration()

    def _inner_next(self):
        from .common.async_utils import MyStopIteration
        try:
            return next(self._result_generator)
        except StopIteration:
            raise MyStopIteration()

    async def aclose(self):
        return await self._run_in_executor(self._result_generator.close)

    async def _run_in_executor(self, f, *args):
        import asyncio
        loop = asyncio.get_event_loop()
        next_result = await loop.run_in_executor(self._pool, f, *args)
        return next_result

    async def update_parameters_experimental(self, parameters: list[dict[str, Any]]):
        """
        Update parameters while the UDFs are running.

        :code:`parameters` should be a list of dicts, with one dict for each
        UDF you are running.

        The dicts should only contain items for those parameters you want to
        update.
        """
        logger.debug("ResultGenerator.update_parameters_experimental: %s", parameters)
        return await self._run_in_executor(
            self._result_generator.update_parameters_experimental, parameters
        )

    async def athrow(self, exc: Exception):
        return await self._run_in_executor(self._result_generator.throw, exc)


RunUDFGenType = ResultGenerator
RunUDFGenTypeL = ResultGenerator
RunUDFAGenType = ResultAsyncGenerator
RunUDFAGenTypeL = ResultAsyncGenerator


class Context:
    """
    Context is the main entry point of the LiberTEM API. It contains
    methods for loading datasets, creating analyses on them and running
    them. In the background, instantiating a Context creates a suitable
    executor and spins up a local Dask cluster unless the executor is
    passed to the constructor.

    .. versionchanged:: 0.7.0
        Removed deprecated methods :code:`create_mask_job`, :code:`create_pick_job`

    Parameters
    ----------

    executor : ~libertem.common.executor.JobExecutor or None
        If None, create a local dask.distributed cluster and client using
        :meth:`~libertem.executor.dask.DaskJobExecutor.make_local` with optimal configuration
        for LiberTEM. It uses all cores and compatible GPUs
        on the local system, but is not set as default Dask scheduler to not interfere
        with other uses of Dask.

    plot_class : libertem.viz.base.Live2DPlot
        Default plot class for live plotting.
        Defaults to :class:`libertem.viz.mpl.MPLLive2DPlot`.

        .. versionadded:: 0.7.0

    Attributes
    ----------

    plot_class : libertem.viz.base.Live2DPlot
        Default plot class for live plotting.
        Defaults to :class:`libertem.viz.mpl.MPLLive2DPlot`.

        .. versionadded:: 0.7.0

    Examples
    --------

    >>> ctx = libertem.api.Context()  # doctest: +SKIP

    >>> # Create a Context using an inline executor for debugging
    >>> # See also Context.make_with() for a more convenient interface!
    >>> from libertem.executor.inline import InlineJobExecutor
    >>> debug_ctx = libertem.api.Context(executor=InlineJobExecutor())
    """

    def __init__(
        self,
        executor: Optional[JobExecutor] = None,
        plot_class: Optional['Live2DPlot'] = None,
    ):
        import traceback
        if executor is None:
            executor = self._create_local_executor()
        self.executor = executor
        self._plot_class = plot_class
        self._created_at = traceback.format_stack()
        self._register_at_exit()

    @classmethod
    def make_with(
        cls,
        executor_spec: ExecutorSpecType = 'dask',
        *,
        cpus: Optional[Union[int, Iterable[int]]] = None,
        gpus: Optional[Union[int, Iterable[int]]] = None,
        plot_class: Optional['Live2DPlot'] = None,
        snooze_timeout: Optional[float] = None,
    ) -> 'Context':
        '''
        Create a Context with a specific kind of executor.

        .. versionadded:: 0.9.0

        This simplifies creating a :class:`Context` for a number of common executor
        choices and allows specification of the resources used to create the executor.
        For more fine-grained control of resource allocation create the
        executor manually and pass it to the :class:`Context`.

        See :ref:`executors` for general information on executors.

        .. versionchanged:: 0.12.0

            Prior to version 0.12.0, this function accepted :code:`*args, **kwargs`
            and passed them to the initializer of :class:`Context`. Given that the
            Context accepts only the :code:`plot_class` keyword-argument this was
            hard-coded into this function for backwards-compatibility, enabling
            the addition of the :code:`cpus` and :code:`gpus` parameters.

        Parameters
        ----------

        executor_spec : ExecutorSpecType, optional, by default "dask"
            A string identifier for executor variants:

            "synchronous", "inline":
                Use a single-process, single-threaded
                :class:`~libertem.executor.inline.InlineJobExecutor`
            "threads":
                Use a multi-threaded :class:`~libertem.executor.concurrent.ConcurrentJobExecutor`
            "dask":
                Create a standard :class:`~libertem.executor.dask.DaskJobExecutor` without
                considering any pre-existing Dask schedulers available on the system, similar to
                the default behaviour of :code:`Context()` called with no arguments. Will create
                by default 1 worker per-CPU core and per-GPU detected on the system.
            "dask-integration":
                Use a JobExecutor that is compatible with the currently active Dask scheduler.
                See :func:`~libertem.executor.integration.get_dask_integration_executor` for
                more information.
            "dask-make-default":
                Create a local :code:`dask.distributed` cluster and client
                using :meth:`~libertem.executor.dask.DaskJobExecutor.make_local`.
                The Client will be set as the default Dask scheduler and will
                persist after the LiberTEM Context closes, which is suitable for downstream
                computation using :code:`dask.distributed`.
            "delayed":
                Create a :class:`~libertem.executor.delayed.DelayedJobExecutor` which performs
                computation using `dask.delayed <https://docs.dask.org/en/stable/delayed.html>`_.
                This functionality is highly experimental at this time, see
                :ref:`delayed_udfs` for more information.
            "pipelined":
                Create a :class:`~libertem.executor.pipelined.PipelinedExecutor`,
                which is suitable for multi-process streaming live processing
                using `LiberTEM-live <https://libertem.github.io/LiberTEM-live/>`_
        cpus : int | Iterable[int], optional
            The number of CPU workers to create, where possible. The meaning of
            CPU worker depends on the type of executor created - threaded executors
            will interpret :code:`cpus` to choose the number of threads, while process-based
            executors will interpret the value to choose the number of processes to spawn.
            The iterable form of of the argument is intended contain CPU-id values to
            enable cpu-pinning where the exector and the system support it, but most
            use cases will only require an integer argument. Executors where this
            parameter does not make sense will raise an error if provided.
            When unspecified the default executor behaviour is retained.
        gpus : int | Iterable[int], optional
            Similar to :code:`cpus`, specifies the number of GPU workers to create
            where the executor chosen supports it, else raise an error. The integer
            form of the argument specifies the total number of workers to create,
            assigned according to the implementation of each executor, while
            the iterable form allows assigment of multiple workers to specific GPUs
            by repeating the id of any given GPU in the argument.
            When unspecified the default executor behaviour is retained.
        plot_class : libertem.viz.base.Live2DPlot, optional
            Plot class for live plotting, passed to :class:`Context`.
        snooze_timeout : float, optional
            Activate automatic executor downscaling after :code:`snooze_timeout`
            minutes of inactivity. This can be used to free resources in a shared
            environment, for example. The executor is automatically brought back
            up when used again after snoozing. Currently only supported for the
            :class:`~libertem.executor.dask.DaskJobExecutor`.

        Raises
        ------
        libertem.common.exceptions.ExecutorSpecException :
            for invalid executor choice or unsupported worker specifications

        Returns
        -------
        Instance of :class:`Context` using a new instance of the specified executor.
        '''
        # The following block is temporary until the handling of cpus/gpus args is
        # pushed onto each exector
        cpu_spec = cpus is not None
        gpu_spec = gpus is not None
        has_spec = cpu_spec or gpu_spec
        limited_execs = ('inline', 'synchronous', 'dask-integration', 'delayed')
        cannot_cpus = executor_spec in limited_execs
        if cpu_spec and cannot_cpus:
            raise ExecutorSpecException(f'Executor type {executor_spec} does not support '
                                        'specifying CPU workers at this time')
        cannot_gpus = executor_spec in limited_execs + ('threads',)
        if gpu_spec and cannot_gpus:
            raise ExecutorSpecException(f'Executor type {executor_spec} does not support '
                                        'specifying GPU workers at this time')
        # Delay import here to avoid cupy import overhead
        if has_spec and executor_spec in ('dask', 'dask-make-default', 'pipelined'):
            from libertem.utils.devices import detect
            spec_args = detect()
            if cpu_spec:
                spec_args['cpus'] = cpus
            if gpu_spec:
                if len(spec_args['cudas']) == 0 and gpus:
                    raise ExecutorSpecException('Cannot specify GPU workers as no GPUs detected')
                spec_args['cudas'] = gpus
        spec = None
        if has_spec and executor_spec in ('dask', 'dask-make-default'):
            spec = cluster_spec(**spec_args)

        if snooze_timeout is not None:
            snooze_timeout *= 60

        executor: JobExecutor
        if executor_spec in ('synchronous', 'inline'):
            executor = InlineJobExecutor()
        elif executor_spec == 'threads':
            n_threads = cpus
            try:
                n_threads = len(n_threads)
            except TypeError:
                pass
            executor = ConcurrentJobExecutor.make_local(n_threads=n_threads)
        elif executor_spec == 'dask':
            executor = DaskJobExecutor.make_local(spec=spec, snooze_timeout=snooze_timeout)
        elif executor_spec == 'dask-integration':
            executor = get_dask_integration_executor()
        elif executor_spec == 'dask-make-default':
            executor = DaskJobExecutor.make_local(
                spec=spec,
                client_kwargs={"set_as_default": True}
            )
        elif executor_spec == 'delayed':
            executor = DelayedJobExecutor()
        elif executor_spec == 'pipelined':
            if has_spec:
                spec = PipelinedExecutor.make_spec(**spec_args)
                executor = PipelinedExecutor(spec=spec)
            else:
                executor = PipelinedExecutor.make_local()
        else:
            raise ExecutorSpecException(
                f'Argument `executor_spec` is {executor_spec}. Allowed are '
                f'"synchronous", "inline", "threads", "dask", "dask-integration",'
                f'"dask-make-default" "delayed" or "pipelined".'
            )
        return cls(executor=executor, plot_class=plot_class)

    @property
    def plot_class(self) -> type['Live2DPlot']:
        if self._plot_class is None:
            from libertem.viz.mpl import MPLLive2DPlot
            self._plot_class = MPLLive2DPlot
        return self._plot_class

    @plot_class.setter
    def plot_class(self, value: type['Live2DPlot']):
        self._plot_class = value

    def load(self, filetype: str, *args, io_backend=None, **kwargs) -> DataSet:
        """
        Load a :class:`~libertem.io.dataset.base.DataSet`. As it doesn't load
        the whole data into RAM at once, you can load and process datasets
        that are bigger than your available RAM. Using fast storage (i.e.
        SSD) is advisable.

        .. versionchanged:: 0.5.0
            Added support for filetype="auto"

        .. versionchanged:: 0.6.0
            Added support for specifying the I/O backend

        Parameters
        ----------
        filetype : str
            one of: %(types)s; or use "auto" to automatically determine filetype and parameters
        io_backend : IOBackend or None
            Use a different I/O backend for this data set
        args
            passed on to the DataSet implementation
        kwargs
            passed on to the DataSet implementation

        Returns
        -------
        DataSet : libertem.io.dataset.base.DataSet
            The loaded dataset

        Note
        ----

        Additional parameters are passed to the concrete
        :class:`~libertem.io.dataset.base.DataSet` implementation.

        Note
        ----
        See :ref:`dataset api` for format-specific documentation.

        Examples
        --------

        Load a data set from a given path, automatically determinig the type:

        >>> ds = ctx.load("auto", path="...")  # doctest: +SKIP

        To configure an alternative I/O backend, in this case configuring
        the mmap backend to enable readahead hints:

        >>> from libertem.io.dataset.base import MMapBackend
        >>> io_backend = MMapBackend(enable_readahead_hints=True)
        >>> ds = ctx.load("auto", path="...", io_backend=io_backend)  # doctest: +SKIP
        """
        # delegate to libertem.io.dataset.load:
        return load(
            filetype,
            *args,
            io_backend=io_backend,
            executor=self.executor,
            enable_async=False,
            **kwargs,
        )

    # If people run with -OO, which strips docstrings, we must not
    # try to treat load.__doc__ as `str`:
    if load.__doc__ is not None:
        load.__doc__ = load.__doc__ % {"types": ", ".join(filetypes.keys())}

    def create_mask_analysis(self, factories: MaskFactoriesType, dataset: DataSet,
            use_sparse: bool = None, mask_count: int = None, mask_dtype: np.dtype = None,
            dtype: np.dtype = None) -> MasksAnalysis:
        """
        Create a mask application analysis. Each factory function should, when
        called, return a numpy array with the same shape as frames in the
        dataset (so :code:`dataset.shape.sig`).

        This is a more high-level interface than
        :class:`~libertem.udf.masks.ApplyMasksUDF` and differs in the way the
        result is returned. With :class:`~libertem.udf.masks.ApplyMasksUDF`, it
        is a single numpy array, here we split it up for each mask we apply,
        make some default visualization available etc.

        Parameters
        ----------
        factories : Union[Callable[[], array_like], Iterable[Callable[[], array_like]]]
            Function or list of functions that take no arguments
            and create masks. The returned masks can be numpy arrays,
            scipy.sparse or sparse https://sparse.pydata.org/ matrices. The mask
            factories should not reference large objects because they can create
            significant overheads when they are pickled and unpickled. If a
            single function is specified, the first dimension is interpreted as
            the mask index.
        dataset : libertem.io.dataset.base.DataSet
            dataset to work on
        use_sparse : bool or None
            * None (default): Use sparse matrix multiplication if all factory functions return a \
            sparse mask, otherwise convert all masks to dense matrices and use dense matrix \
            multiplication
            * True: Convert all masks to sparse matrices.
            * False: Convert all masks to dense matrices.
        mask_count : int, optional
            Specify the number of masks if a single factory function is
            used so that the number of masks can be determined without calling
            the factory function.
        mask_dtype : numpy.dtype, optional
            Specify the dtype of the masks so that mask dtype can be determined without
            calling the mask factory functions. This can be used to override the
            mask dtype in the result dtype determination. As an example, setting
            this to :code:`np.float32` means that returning masks of type float64 will not switch
            the calculation and result dtype to float64 or complex128.
        dtype : numpy.dtype, optional
            Specify the dtype to do the calculation in.
            Integer dtypes are possible if the numpy casting rules allow this
            for source and mask data.

        Returns
        -------
        MasksAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.masks.MasksResultSet`.

        Examples
        --------

        >>> # Use intermediate variables instead of referencing
        >>> # large complex objects like a dataset within the
        >>> # factory function
        >>> shape = dataset.shape.sig
        >>> analysis = ctx.create_mask_analysis(
        ...     factories=[lambda: np.ones(shape)],
        ...     dataset=dataset
        ... )
        >>> result = ctx.run(analysis)
        >>> result.mask_0.raw_data.shape
        (16, 16)
        """
        return MasksAnalysis(
            dataset=dataset,
            parameters={
                "factories": factories,
                "use_sparse": use_sparse,
                "mask_count": mask_count,
                "mask_dtype": mask_dtype,
                "dtype": dtype},
        )

    def create_com_analysis(self, dataset: DataSet, cx: int = None, cy: int = None,
                            mask_radius: float = None, flip_y: bool = False,
                            mask_radius_inner: float = None,
                            scan_rotation: float = 0.0) -> COMAnalysis:
        """
        Create a center-of-mass (first moment) analysis, possibly masked.

        Parameters
        ----------
        dataset
            the dataset to work on
        cx
            reference center x value
        cy
            reference center y value
        mask_radius
            mask out intensity outside of `mask_radius` from `(cy, cx)`
        mask_radius_inner
            mask out intensity except for the ring between `mask_radius_inner` and
            `mask_radius`, centered around `(cy, cx)`

            .. versionadded:: 0.8.0
        flip_y : bool
            Flip the Y coordinate. Some detectors, namely Quantum Detectors Merlin,
            may have pixel (0, 0) at the lower left corner. This has to be corrected
            to get the sign of the y shift as well as curl and divergence right.

            .. versionadded:: 0.6.0

        scan_rotation : float
            Scan rotation in degrees.
            The optics of an electron microscope can rotate the image. Furthermore, scan
            generators may allow scanning in arbitrary directions. This means that the x and y
            coordinates of the detector image are usually not parallel to the x and y scan
            coordinates. For interpretation of center of mass shifts, however, the shift vector
            in detector coordinates has to be put in relation to the position on the sample.
            The :code:`scan_rotation` parameter can be used to rotate the detector coordinates
            to match the scan coordinate system. A positive value rotates the displacement
            vector clock-wise. That means if the detector seems rotated to the right relative
            to the scan, this value should be negative to counteract this rotation.

            .. versionadded:: 0.6.0

        Returns
        -------
        COMAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.com.COMResultSet`.
        """
        if dataset.shape.nav.dims != 2:
            raise ValueError("incompatible dataset: need two navigation dimensions")
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {
            name: loc[name]
            for name in ['cx', 'cy', 'flip_y', 'scan_rotation']
            if loc[name] is not None
        }
        if mask_radius is not None:
            parameters['r'] = mask_radius
        if mask_radius_inner is not None:
            if mask_radius is None:
                raise ValueError(
                    "incompatible parameters: must pass both `mask_radius` and "
                    "`mask_radius_inner` for annular CoM"
                )
            parameters['ri'] = mask_radius_inner
        analysis = COMAnalysis(
            dataset=dataset, parameters=parameters
        )
        return analysis

    def create_radial_fourier_analysis(self, dataset: DataSet, cx: float = None, cy: float = None,
            ri: float = None, ro: float = None, n_bins: int = None, max_order: int = None,
            use_sparse: bool = None) -> RadialFourierAnalysis:
        """
        Create an Analysis that calculates the Fourier transform of rings around the center.

        See :ref:`radialfourier app` for details on the method!

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
        n_bins
            number of bins
        max_order
            maximum order of calculated Fourier component

        Returns
        -------
        RadialFourierAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.radialfourier.RadialFourierResultSet`.
        """
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {
            name: loc[name]
            for name in ['cx', 'cy', 'ri', 'ro', 'n_bins', 'max_order', 'use_sparse']
            if loc[name] is not None
        }
        analysis = RadialFourierAnalysis(
            dataset=dataset, parameters=parameters
        )
        return analysis

    def create_disk_analysis(self, dataset: DataSet, cx: int = None, cy: int = None,
                             r: int = None) -> DiskMaskAnalysis:
        """
        Create an Analysis that integrates over a disk (i.e. filled circle).

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

        Returns
        -------
        DiskMaskAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.masks.SingleMaskResultSet`.
        """
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {name: loc[name] for name in ['cx', 'cy', 'r'] if loc[name] is not None}
        return DiskMaskAnalysis(
            dataset=dataset, parameters=parameters
        )

    def create_ring_analysis(self, dataset: DataSet, cx: int = None, cy: int = None,
                             ri: int = None, ro: int = None) -> RingMaskAnalysis:
        """
        Create an Analysis that integrates over a ring.

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

        Returns
        -------
        RingMaskAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.masks.SingleMaskResultSet`.
        """
        if dataset.shape.sig.dims != 2:
            raise ValueError("incompatible dataset: need two signal dimensions")
        loc = locals()
        parameters = {name: loc[name] for name in ['cx', 'cy', 'ri', 'ro'] if loc[name] is not None}
        return RingMaskAnalysis(
            dataset=dataset, parameters=parameters
        )

    def create_point_analysis(self, dataset: DataSet, x: int = None,
                              y: int = None) -> PointMaskAnalysis:
        """
        Create an Analysis that selects the pixel with coords (y, x) from each frame

        Returns
        -------
        PointMaskAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.masks.SingleMaskResultSet`.
        """
        if dataset.shape.nav.dims > 2:
            raise ValueError("incompatible dataset: need at most two navigation dimensions")
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

    def create_sum_analysis(self, dataset) -> SumAnalysis:
        """
        Create an Analysis that sums all signal elements along the navigation
        dimension, preserving the signal dimension.

        Parameters
        ----------
        dataset
            the dataset to work on

        Returns
        -------
        SumAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.sum.SumResultSet`.
        """
        return SumAnalysis(dataset=dataset, parameters={})

    def create_pick_analysis(self, dataset: DataSet, x: int, y: int = None,
                             z: int = None) -> PickFrameAnalysis:
        """
        Create an Analysis that picks a single frame / signal element from (z, y, x).
        The number of parameters must match number of navigation dimensions in the dataset,
        for example if you have a 4D dataset with two signal dimensions and two navigation
        dimensions, you need to specify x and y.

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
        PickFrameAnalysis : libertem.analysis.base.Analysis
            When run by the Context, this Analysis generates a
            :class:`libertem.analysis.raw.PickResultSet`.

        Examples
        --------

        >>> dataset = ctx.load(
        ...     filetype="memory",
        ...     data=np.zeros([16, 16, 16, 16, 16], dtype=np.float32),
        ...     sig_dims=2
        ... )
        >>> analysis = ctx.create_pick_analysis(dataset=dataset, x=9, y=8, z=7)
        >>> result = ctx.run(analysis)
        >>> assert result.intensity.raw_data.shape == tuple(dataset.shape.sig)
        """
        loc = locals()
        parameters = {name: loc[name] for name in ['x', 'y', 'z'] if loc[name] is not None}
        return PickFrameAnalysis(dataset=dataset, parameters=parameters)

    def run(
        self, job: Analysis,
        roi: RoiT = None,
        progress: Union[bool, ProgressReporter] = False,
        corrections: Optional[CorrectionSet] = None,
    ) -> Union[np.ndarray, AnalysisResultSet]:
        """
        Run the given :class:`~libertem.analysis.base.Analysis`
        and return the result data.

        .. versionchanged:: 0.5.0
            Added the :code:`progress` parameter

        .. versionchanged:: 0.6.0
            Added the :code:`corrections` parameter

        .. versionchanged:: 0.7.0
            Removed deprecated Job support, now only UDF-based analyses are supported

        Parameters
        ----------
        job
            the analysis to run
        roi : numpy.ndarray, sparse array or coordinate tuple(s), optional
            Boolean mask of the navigation dimension. See :ref:`udf roi`.
        progress : bool | ProgressReporter
            Show progress bar. Toggle with boolean flag or supply instance of
            :class:`libertem.common.progress.ProgressReporter` for custom
            handling of progress display.
        corrections
            Corrections to apply, i.e. dark frame substraction, applying a gain map, ...

        Returns
        -------
        result : libertem.analysis.base.AnalysisResultSet
            Running an Analysis returns a :class:`libertem.analysis.base.AnalysisResultSet`.
            See the matching :code:`create_*_analysis` function for documentation
            of the specific :code:`AnalysisResultSet` subclass or :class:`numpy.ndarray` that
            is being returned.
        """
        analysis = job  # keep the old kwarg name for backward-compat.
        if roi is None:
            roi = analysis.get_roi()
        else:
            roi = sparse_to_coo(roi, analysis.dataset.shape.nav)
        udf_results: UDFResultDict = self.run_udf(  # type:ignore[assignment]
            dataset=analysis.dataset, udf=analysis.get_udf(), roi=roi,
            corrections=corrections, progress=progress,
        )
        # Here we plot only after the computation is completed, meaning the damage should be
        # the ROI or the entire nav dimension.
        # TODO live plotting following libertem.web.jobs.JobDetailHandler.run_udf
        # Current Analysis interface possibly made obsolete by #1013, so deferred
        damage: "nt.ArrayLike"
        if roi is None:
            damage = True
        else:
            damage = to_dense(roi)
        return analysis.get_udf_results(udf_results, roi, damage=damage)

    def run_udf(
        self,
        dataset: DataSet,
        udf: Union[UDF, Iterable[UDF]],
        roi: RoiT = None,
        corrections: Optional[CorrectionSet] = None,
        progress: Union[bool, ProgressReporter] = False,
        backends=None,
        plots=None,
        sync=True,
    ) -> Union[RunUDFResultType, RunUDFSyncL, RunUDFAsync, RunUDFAsyncL]:
        """
        Run :code:`udf` on :code:`dataset`, restricted to the region of interest :code:`roi`.

        .. versionchanged:: 0.5.0
            Added the :code:`progress` parameter

        .. versionchanged:: 0.6.0
            Added the :code:`corrections` and :code:`backends` parameter

        .. versionchanged:: 0.7.0
            Added the :code:`plots` and :code:`sync` parameters,
            and the ability to run multiple UDFs on the same data in a single pass.

        Raises
        ------
        UDFRunCancelled
            Either the run was cancelled using :meth:`AsyncJobExecutor.cancel`,
            or the underlying data source was interrupted.

        Parameters
        ----------
        dataset
            The dataset to work on

        udf
            UDF instance you want to run, or a list of UDF instances

        roi : numpy.ndarray, sparse array or coordinate tuple(s), optional
            Region of interest as bool mask over the navigation axes of the dataset.
            See :ref:`udf roi`.

        progress : bool | ProgressReporter
            Show progress bar. Toggle with boolean flag or supply instance of
            :class:`libertem.common.progress.ProgressReporter` for custom
            handling of progress display.

        corrections
            Corrections to apply while running the UDF. If none are given,
            the corrections that are part of the :code:`DataSet` are used,
            if there are any. See also :ref:`corrections`.

        backends : None or iterable containing 'numpy', 'cupy' and/or 'cuda'
            Restrict the back-end to a subset of the capabilities of the UDF.
            This can be useful for testing hybrid UDFs.

        plots : None or True or List[List[Union[str, Tuple[str, Callable]]]] or List[LivePlot]
            - :code:`None`: don't plot anything (default)
            - :code:`True`: plot all 2D UDF result buffers
            - :code:`List[List[...]]`: plot the named UDF buffers. Pass a list of names or
              (name, callable) tuples for each UDF you want to plot. If the callable is specified,
              it is applied to the UDF buffer before plotting.
            - :code:`List[LivePlot]`: :class:`~libertem.viz.base.LivePlot` instance for each
              channel you want to plot

            .. versionadded:: 0.7.0

        sync : bool
            By default, `run_udf` is a synchronous method. If `sync` is set to `False`,
            it is awaitable instead.

            .. versionadded:: 0.7.0

        Returns
        -------
        dict or Tuple[dict, ...]
            Return value of the UDF containing the result buffers of
            type :class:`libertem.common.buffers.BufferWrapper`. Note that a
            :class:`~libertem.common.buffers.BufferWrapper` can be used like
            a :class:`numpy.ndarray` in many cases because it implements
            :meth:`__array__`. You can access the underlying numpy array using the
            :attr:`~libertem.common.buffers.BufferWrapper.data` property.

            If a list of UDFs was passed in, the returned type is
            a Tuple[dict[str,BufferWrapper], ...].

        Examples
        --------
        Run the :class:`~libertem.udf.sum.SumUDF` on a data set:

        >>> from libertem.udf.sum import SumUDF
        >>> result = ctx.run_udf(dataset=dataset, udf=SumUDF())
        >>> np.array(result["intensity"]).shape
        (32, 32)
        >>> # intensity is the name of the result buffer, defined in the SumUDF

        Running a UDF on a subset of data:

        >>> from libertem.udf.sumsigudf import SumSigUDF
        >>> roi = dataset.roi[0, 0]
        >>> result = ctx.run_udf(dataset=dataset, udf=SumSigUDF(), roi=roi)
        >>> # to get the full navigation-shaped results, with NaNs where the `roi` was False:
        >>> np.array(result["intensity"]).shape
        (16, 16)
        >>> # to only get the selected results as a flat array:
        >>> result["intensity"].raw_data.shape
        (1,)
        """
        # TODO: add a more narrow type signature - instead of a Union[...], we should
        # have overloads depending on both the type of `udf` and the `Literal[...]` value
        # of `iterate`. This was not yet added because of
        # https://github.com/python/mypy/issues/6580
        # In short, we can't have an overload `run_udf(..., plots=None, sync: Literal[True])`
        # because either we have a non-default argument after a default argument, or we have
        # `Literal[True] = ...` which overlaps with `Literal[False] = ...``
        with tracer.start_as_current_span("Context.run_udf"):
            if sync:
                return self._run_sync(
                    dataset=dataset,
                    udf=udf,
                    roi=roi,
                    corrections=corrections,
                    progress=progress,
                    backends=backends,
                    plots=plots,
                    iterate=False,
                )
            else:
                return self._run_async(
                    dataset=dataset,
                    udf=udf,
                    roi=roi,
                    corrections=corrections,
                    progress=progress,
                    backends=backends,
                    plots=plots,
                    iterate=False,
                )

    def run_udf_iter(
        self,
        dataset: DataSet,
        udf: Union[UDF, Iterable[UDF]],
        roi: RoiT = None,
        corrections: CorrectionSet = None,
        progress: Union[bool, ProgressReporter] = False,
        backends=None,
        plots=None,
        sync=True,
    ) -> Union[RunUDFGenType, RunUDFAGenType, RunUDFGenTypeL, RunUDFAGenTypeL]:
        """
        Run :code:`udf` on :code:`dataset`, restricted to the region of interest :code:`roi`.
        Yields partial results after each merge operation.

        .. versionadded:: 0.7.0

        Parameters
        ----------
        dataset
            The dataset to work on

        udf
            UDF instance you want to run, or a list of UDF instances

        roi : numpy.ndarray, sparse array or coordinate tuple(s), optional
            Region of interest as bool mask over the navigation axes of the dataset.
            See :ref:`udf roi`.

        progress : bool | ProgressReporter
            Show progress bar. Toggle with boolean flag or supply instance of
            :class:`libertem.common.progress.ProgressReporter` for custom
            handling of progress display.

        corrections
            Corrections to apply while running the UDF. If none are given,
            the corrections that are part of the :code:`DataSet` are used,
            if there are any. See also :ref:`corrections`.

        backends : None or iterable containing 'numpy', 'cupy' and/or 'cuda'
            Restrict the back-end to a subset of the capabilities of the UDF.
            This can be useful for testing hybrid UDFs.

        plots : None or True or List[List[Union[str, Tuple[str, Callable]]]] or List[LivePlot]
            - :code:`None`: don't plot anything (default)
            - :code:`True`: plot all 2D UDF result buffers
            - :code:`List[List[...]]`: plot the named UDF buffers. Pass a list of names or
              (name, callable) tuples for each UDF you want to plot. If the callable is specified,
              it is applied to the UDF buffer before plotting.
            - :code:`List[LivePlot]`: :class:`~libertem.viz.base.LivePlot` instance for each
              channel you want to plot

        sync : bool
            By default, `run_udf_iter` is a synchronous method. If `sync` is set to `False`,
            an async generator will be returned instead.

        Returns
        -------
        Generator[UDFResults]
            Generator of :class:`~libertem.udf.base.UDFResults` container objects.
            Their attribute :code:`buffers` is the list of result buffer dictionaries for the UDFs.
            Attribute :code:`damage` is a :class:`~libertem.common.buffers.BufferWrapper`
            of :code:`kind='nav'`, :code:`dtype=bool` indicating the positions
            in nav space that have been processed already.

        Examples
        --------
        Run the :class:`~libertem.udf.sum.SumUDF` on a data set:

        >>> from libertem.udf.sum import SumUDF
        >>> for result in ctx.run_udf_iter(dataset=dataset, udf=SumUDF()):
        ...     assert np.array(result.buffers[0]["intensity"]).shape == (32, 32)
        >>> np.array(result.buffers[0]["intensity"]).shape
        (32, 32)
        >>> # intensity is the name of the result buffer, defined in the SumUDF
        """
        if sync:
            return self._run_sync(
                dataset=dataset,
                udf=udf,
                roi=roi,
                corrections=corrections,
                progress=progress,
                backends=backends,
                plots=plots,
                iterate=True,
            )
        else:
            return self._run_async(
                dataset=dataset,
                udf=udf,
                roi=roi,
                corrections=corrections,
                progress=progress,
                backends=backends,
                plots=plots,
                iterate=True,
            )

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: UDF,
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: Literal[False],
    ) -> UDFResultDict:
        ...

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: UDF,
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: Literal[True],
    ) -> RunUDFGenType: ...

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: Iterable[UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: Literal[False],
    ) -> RunUDFSyncL: ...

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: Iterable[UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: Literal[True],
    ) -> RunUDFGenTypeL: ...

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: Union[Iterable[UDF], UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: Literal[True],
    ) -> Union[RunUDFGenType, RunUDFGenTypeL]:
        ...

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: Union[Iterable[UDF], UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: Literal[False],
    ) -> Union[UDFResultDict, RunUDFSyncL]:
        ...

    @overload
    def _run_sync(
        self,
        dataset: DataSet,
        udf: Union[Iterable[UDF], UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends: Optional[Any],
        plots: Optional[Any],
        iterate: bool,
    ) -> Union[UDFResultDict, RunUDFSyncL, RunUDFGenType, RunUDFGenTypeL]: ...

    def _run_sync(
        self,
        dataset: DataSet,
        udf: Union[UDF, Iterable[UDF]],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: bool,
    ):
        """
        Run the given UDF(s), either returning the final result (when
        :code:`iterate=False` is given), or a generator that yields partial results.
        """
        enable_plotting = bool(plots)

        udf_is_list = isinstance(udf, Iterable)
        if not isinstance(udf, Iterable):  # duplicate, because mypy.
            udfs = [udf]
        else:
            udfs = list(udf)

        if len(udfs) == 0:
            raise ValueError("empty list of UDFs - nothing to do!")

        if enable_plotting:
            with tracer.start_as_current_span("prepare_plots"):
                plots = self._prepare_plots(udfs, dataset, roi, plots)

        if corrections is None:
            corrections = dataset.get_correction_data()

        if roi is not None:
            roi = sparse_to_coo(roi, dataset.shape.nav)
            if roi.dtype is not np.dtype(bool):
                warnings.warn(f"ROI dtype is {roi.dtype}, expected bool. Attempting cast to bool.")
                roi = roi.astype(bool)

        progress_reporter: Optional[ProgressReporter]
        if isinstance(progress, ProgressReporter):
            progress_reporter = progress
            progress = True
        else:
            progress_reporter = None

        def _run_sync_wrap() -> ResultGenerator:
            runner_cls = self.executor.get_udf_runner()
            runner = runner_cls(
                udfs,
                progress_reporter=progress_reporter,
            )
            result_iter = runner.run_for_dataset_sync(
                dataset=dataset,
                executor=self.executor,
                roi=roi,
                progress=progress,
                corrections=corrections,
                backends=backends,
                iterate=(iterate or enable_plotting)
            )

            def _inner():
                for udf_results in result_iter:
                    yield udf_results
                    if enable_plotting:
                        self._update_plots(
                            plots, udfs, udf_results.buffers, udf_results.damage.data, force=False
                        )
                if enable_plotting:
                    self._update_plots(
                        plots, udfs, udf_results.buffers, udf_results.damage.data, force=True
                    )
            return ResultGenerator(_inner(), runner=runner, result_iter=result_iter)

        if iterate:
            return _run_sync_wrap()
        else:
            udf_results = run_gen_get_last(_run_sync_wrap())
            if udf_is_list:
                return udf_results.buffers
            else:
                return udf_results.buffers[0]

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: UDF,
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: Literal[False],
    ) -> RunUDFAsync: ...

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: Iterable[UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: Literal[False],
    ) -> RunUDFAsyncL: ...

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: UDF,
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: Literal[True],
    ) -> RunUDFAGenType: ...

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: Iterable[UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: Literal[True],
    ) -> RunUDFAGenTypeL: ...

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: Union[UDF, Iterable[UDF]],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: Literal[True],
    ) -> Union[RunUDFAGenTypeL, RunUDFAGenType]: ...

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: Union[UDF, Iterable[UDF]],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: Literal[False],
    ) -> Union[RunUDFAsync, RunUDFAsyncL]: ...

    @overload
    def _run_async(
        self,
        dataset: DataSet,
        udf: Union[Iterable[UDF], UDF],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: bool,
    ) -> Union[RunUDFAsync, RunUDFAsyncL, RunUDFAGenType, RunUDFAGenTypeL]: ...

    def _run_async(
        self,
        dataset: DataSet,
        udf: Union[UDF, Iterable[UDF]],
        roi: RoiT,
        corrections: Optional[CorrectionSet],
        progress: Union[bool, ProgressReporter],
        backends,
        plots,
        iterate: bool,
    ):
        """
        Wraps :code:`_run_sync` into an asynchronous generator,
        and either returns the generator itself, or the end result.
        """
        sync_generator = self._run_sync(
            dataset=dataset,
            udf=udf,
            roi=roi,
            corrections=corrections,
            progress=progress,
            backends=backends,
            plots=plots,
            iterate=True,
        )

        async def _run_async_wrap() -> UDFResultDict:
            udf_results = await run_agen_get_last(async_generator(sync_generator))
            return udf_results.buffers[0]

        async def _run_async_wrap_l() -> list[UDFResultDict]:
            udf_results = await run_agen_get_last(async_generator(sync_generator))
            return udf_results.buffers

        if iterate:
            return ResultAsyncGenerator(result_generator=sync_generator)
        else:
            if isinstance(udf, Iterable):
                return _run_async_wrap_l()
            else:
                return _run_async_wrap()

    def _get_default_plot_chans(self, buffers):
        from libertem.viz import get_plottable_2D_channels
        return [
            get_plottable_2D_channels(bufferset)
            for bufferset in buffers
        ]

    def _prepare_plots(
        self,
        udfs: list[UDF],
        dataset: DataSet,
        roi: RoiT,
        plots,
    ) -> list['Live2DPlot']:
        runner_cls = self.executor.get_udf_runner()
        dry_results = runner_cls.dry_run(udfs, dataset, roi)

        # cases to consider:
        # 1) plots is `True`: default plots of all eligible channels
        # 2) plots is List[List[str]] or List[List[(str, callable)]]: set channels from `plots`
        # 3) plots is List[LivePlot]: use customized plots as they are

        channels = None

        # 1) plots is `True`: default plots of all eligible channels
        if plots is True:
            channels = self._get_default_plot_chans(dry_results.buffers)
            for idx, udf in enumerate(udfs):
                if len(channels[idx]) == 0:
                    warnings.warn(
                        f"No plottable channels found for UDF "
                        f"#{idx}: {udf.__class__.__name__}, not plotting."
                    )
        # 2) plots is List[List[str]] or List[List[(str, callable)]]: set channels from `plots`
        elif (isinstance(plots, (list, tuple))
                and all(isinstance(p, (list, tuple)) for p in plots)
                and all(all(isinstance(pp, (str, list, tuple)) for pp in p) for p in plots)):
            channels = plots
        # 3) plots is probably List[LivePlot]: use customized plots as they are
        else:
            return plots

        plots = []
        for idx, (udf, udf_channels) in enumerate(zip(udfs, channels)):
            for channel in udf_channels:
                p0 = self.plot_class(
                    dataset,
                    udf=udf,
                    roi=roi,
                    channel=channel,
                    # Create an UDFResult from this single UDF
                    udfresult=UDFResults(
                        (dry_results.buffers[idx],),
                        dry_results.damage
                    ),
                )
                p0.display()
                plots.append(p0)
        return plots

    def _update_plots(
        self,
        plots: list['Live2DPlot'],
        udfs: list[UDF],
        udf_results: tuple[dict[str, BufferWrapper], ...],
        damage: np.ndarray,
        force: bool = False,
    ):
        for plot in plots:
            udf = plot.get_udf()
            udf_index = udfs.index(udf)
            plot.new_data(udf_results[udf_index], damage, force=force)

    def display(self, dataset: DataSet, udf: UDF, roi: RoiT = None):
        """
        Show information about the UDF in combination with the given DataSet.
        """
        import html

        class _UDFInfo:
            def __init__(self, title, buffers):
                self.title = title
                self.buffers = buffers

            def _repr_html_(self):
                def _e(obj):
                    return html.escape(str(obj))

                rows = [
                    "<tr>"
                    f"<td>{_e(key)}</td>"
                    f"<td>{_e(buf.kind)}</td>"
                    f"<td>{_e(buf.extra_shape)}</td>"
                    f"<td>{_e(buf.shape)}</td>"
                    f"<td>{_e(buf.dtype)}</td>"
                    "</tr>"
                    for key, buf in self.buffers.items()
                    if buf.use != "private"
                ]
                rows = "\n".join(rows)
                general = f"""
                <table>
                    <tbody>
                        <tr>
                            <th>Processing method</th>
                            <td>{_e(udf.get_method())}</td>
                        </tr>
                        <tr>
                            <th>Compute Backends</th>
                            <td>{_e(" ,".join(udf.get_backends()))}</td>
                        </tr>
                        <tr>
                            <th>Preferred input dtype</th>
                            <td>{_e(np.dtype(udf.get_preferred_input_dtype()))}</td>
                        </tr>
                    </tbody>
                </table>
                """
                return f"""
                <h2>{_e(self.title)}</h2>
                <h3>General</h3>
                {general}
                <h3>Result types</h3>
                <p>Note: these may vary with different data sets</p>
                <table>
                    <thead>
                        <th>Name</th>
                        <th>Kind</th>
                        <th>Extra Shape</th>
                        <th>Concrete Shape</th>
                        <th>dtype</th>
                    </thead>
                    <tbody>
                    {rows}
                    </tbody>
                </table>
                """
        runner_cls = self.executor.get_udf_runner()

        if roi is not None:
            roi = sparse_to_coo(roi, dataset.shape.nav)

        return _UDFInfo(
            title=udf.__class__.__name__,
            buffers=runner_cls.inspect_udf(udf, dataset, roi),
        )

    def map(self, dataset: DataSet, f, roi: RoiT = None,
            progress: Union[bool, ProgressReporter] = False,
            corrections: CorrectionSet = None,
            backends=None) -> BufferWrapper:
        '''
        Create an :class:`AutoUDF` with function :meth:`f` and run it on :code:`dataset`

        .. versionchanged:: 0.5.0
            Added the :code:`progress` parameter

        .. versionchanged:: 0.6.0
            Added the :code:`corrections` and :code:`backends` parameter

        Parameters
        ----------

        dataset:
            The dataset to work on
        f:
            Function that accepts a frame as the only parameter. It should return a strongly
            reduced output compared to the size of a frame.
        roi : numpy.ndarray, sparse array or coordinate tuple(s), optional
            Region of interest as bool mask over the navigation axes of the dataset.
            See :ref:`udf roi`.
        progress : bool | ProgressReporter
            Show progress bar. Toggle with boolean flag or supply instance of
            :class:`libertem.common.progress.ProgressReporter` for custom
            handling of progress display.
        corrections
            Corrections to apply while running the function. If none are given,
            the corrections that are part of the :code:`DataSet` are used,
            if there are any. See also :ref:`corrections`.
        backends : None or iterable containing 'numpy', 'cupy' and/or 'cuda'
            Restrict the back-end to a subset of the capabilities of the UDF.
            This can be useful for testing hybrid UDFs.

        Returns
        -------

        BufferWrapper : libertem.common.buffers.BufferWrapper
            The result of the UDF. Access the underlying numpy array using the
            :attr:`~libertem.common.buffers.BufferWrapper.data` property.
            Shape and dtype is inferred automatically from :code:`f`.
        '''
        udf = AutoUDF(f=f)
        results: UDFResultDict = self.run_udf(  # type:ignore[assignment]
            dataset=dataset,
            udf=udf,
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
        )
        return results['result']

    def _create_local_executor(self):
        return DaskJobExecutor.make_local()

    def close(self):
        self.executor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _register_at_exit(self):
        """
        Register at-exit handler, to make sure the executor is closed
        """
        weak_ctx = weakref.ref(self)

        def _exit():
            if weak_ctx() is None:
                return
            try:
                weak_ctx().close()
            except Exception:  # pragma: no cover
                # can't be sure that logging is still alive,
                # so we have to fall back to a normal print here:
                print("\n\n\n\n\n")
                print("Exception in atexit handler for Context created here:")
                print("".join(weak_ctx()._created_at))
                print("\n\n\n\n\n")
                raise

        atexit.register(_exit)

    def export_dataset(
        self,
        dataset: DataSet,
        *,
        path: os.PathLike,
        progress: bool = False,
        overwrite: bool = False,
    ):
        """
        Export the dataset to another format on disk

        At this time can only convert to numpy *.npy* format,
        but future extensions are possible.

        The written data will have any reshaping / sync_offset
        properties of the DataSet effectively baked into
        the new file.

        .. note::
            All workers used by the Context must have access to the
            same file path, over a network file system if necessary,
            in order to correctly save data to the file.

        .. versionadded:: 0.12.0

        Parameters
        ----------
        dataset : lt.Dataset
            The dataset to save to disk
        path : os.PathLike
            The file path to export the data to, will
            raise ValueError if the suffix is unrecognized
            (currently supports only .npy)
        progress : bool, optional
            Whether to display a progress bar for the export,
            by default False
        overwrite : bool , optional
            If the save path already exists, raise FileExistsError unless
            overwrite is True, by default False

        Raises
        ------
        ValueError
            If the path suffix is not supported
        FileExistsError
            If overwrite is True and the save path exists
        """
        path = pathlib.Path(path)
        if path.suffix != '.npy':
            raise ValueError(
                f'Unrecognized file extension {path.suffix} '
                'only .npy is currently supported.'
            )

        if not overwrite and path.is_file():
            raise FileExistsError(
                f'Cannot export dataset to existing path {path} .'
                'Use overwrite=True to force export.'
            )

        from libertem.udf.record import RecordUDF
        udf = RecordUDF(path)
        self.run_udf(dataset, udf, progress=progress)
