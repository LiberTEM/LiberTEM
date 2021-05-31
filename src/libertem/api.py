from typing import Union, Dict, Iterable, Generator, Coroutine, AsyncGenerator
import warnings

import numpy as np
from libertem.corrections import CorrectionSet
from libertem.io.dataset import load, filetypes
from libertem.io.dataset.base import DataSet
from libertem.common.buffers import BufferWrapper
from libertem.executor.dask import DaskJobExecutor
from libertem.executor.base import JobExecutor
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
from libertem.udf.base import UDFRunner, UDF, UDFResults
from libertem.udf.auto import AutoUDF
from libertem.utils.async_utils import async_generator


RunUDFResultType = Dict[str, BufferWrapper]
RunUDFAsync = Coroutine[RunUDFResultType, None, None]
RunUDFGenType = Generator[RunUDFResultType, None, None]
RunUDFAGenType = AsyncGenerator[RunUDFResultType, None]


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

    executor : ~libertem.executor.base.JobExecutor or None
        If None, create a
        :class:`~libertem.executor.dask.DaskJobExecutor` that uses all cores
        on the local system.
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
    >>> from libertem.executor.inline import InlineJobExecutor
    >>> debug_ctx = libertem.api.Context(executor=InlineJobExecutor())
    """

    def __init__(self, executor: JobExecutor = None, plot_class=None):
        if executor is None:
            executor = self._create_local_executor()
        if not isinstance(executor, JobExecutor):
            raise ValueError(
                f'Argument `executor` is not an instance of {JobExecutor}, '
                f'got type "{type(executor)}" instead.'
            )
        self.executor = executor
        self._plot_class = plot_class

    @property
    def plot_class(self):
        if self._plot_class is None:
            from libertem.viz.mpl import MPLLive2DPlot
            self._plot_class = MPLLive2DPlot
        return self._plot_class

    @plot_class.setter
    def plot_class(self, value):
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
        return load(filetype, *args, io_backend=io_backend, executor=self.executor, **kwargs)

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
                            mask_radius: int = None, flip_y: bool = False,
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
            mask out intensity outside of mask_radius from (cy, cx)
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
            coordinates. For interpretation of center of mass sifts, however, the shift vector
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
        roi: np.ndarray = None,
        progress: bool = False,
        corrections: CorrectionSet = None,
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
        roi : numpy.ndarray, optional
            Boolean mask of the navigation dimension.
        progress : bool
            Show progress bar
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
        udf_results = self.run_udf(
            dataset=analysis.dataset, udf=analysis.get_udf(), roi=roi,
            progress=progress, corrections=corrections,
        )
        # Here we plot only after the computation is completed, meaning the damage should be
        # the ROI or the entire nav dimension.
        # TODO live plotting following libertem.web.jobs.JobDetailHandler.run_udf
        # Current Analysis interface possibly made obsolete by #1013, so deferred
        if roi is None:
            damage = True
        else:
            damage = roi
        return analysis.get_udf_results(udf_results, roi, damage=damage)

    def run_udf(
            self,
            dataset: DataSet,
            udf: Union[UDF, Iterable[UDF]],
            roi: np.ndarray = None,
            corrections: CorrectionSet = None,
            progress: bool = False,
            backends=None,
            plots=None,
            sync=True,
    ) -> Union[RunUDFResultType, RunUDFAsync]:
        """
        Run :code:`udf` on :code:`dataset`, restricted to the region of interest :code:`roi`.

        .. versionchanged:: 0.5.0
            Added the :code:`progress` parameter

        .. versionchanged:: 0.6.0
            Added the :code:`corrections` and :code:`backends` parameter

        .. versionchanged:: 0.7.0
            Added the :code:`plots` and :code:`sync` parameters,
            and the ability to run multiple UDFs on the same data in a single pass.

        Parameters
        ----------
        dataset
            The dataset to work on

        udf
            UDF instance you want to run, or a list of UDF instances

        roi : numpy.ndarray
            Region of interest as bool mask over the navigation axes of the dataset

        progress : bool
            Show progress bar

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
        dict or Tuple[dict]
            Return value of the UDF containing the result buffers of
            type :class:`libertem.common.buffers.BufferWrapper`. Note that a
            :class:`~libertem.common.buffers.BufferWrapper` can be used like
            a :class:`numpy.ndarray` in many cases because it implements
            :meth:`__array__`. You can access the underlying numpy array using the
            :attr:`~libertem.common.buffers.BufferWrapper.data` property.

            If a list of UDFs was passed in, the returned type is
            a Tuple[dict[str,BufferWrapper]].

        Examples
        --------
        Run the `SumUDF` on a data set:

        >>> from libertem.udf.sum import SumUDF
        >>> result = ctx.run_udf(dataset=dataset, udf=SumUDF())
        >>> np.array(result["intensity"]).shape
        (32, 32)
        >>> # intensity is the name of the result buffer, defined in the SumUDF

        Running a UDF on a subset of data:

        >>> from libertem.udf.sumsigudf import SumSigUDF
        >>> roi = np.zeros(dataset.shape.nav, dtype=bool)
        >>> roi[0, 0] = True
        >>> result = ctx.run_udf(dataset=dataset, udf=SumSigUDF(), roi=roi)
        >>> # to get the full navigation-shaped results, with NaNs where the `roi` was False:
        >>> np.array(result["intensity"]).shape
        (16, 16)
        >>> # to only get the selected results as a flat array:
        >>> result["intensity"].raw_data.shape
        (1,)
        """

        fn = self._run_async
        if sync:
            fn = self._run_sync

        return fn(
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
            roi: np.ndarray = None,
            corrections: CorrectionSet = None,
            progress: bool = False,
            backends=None,
            plots=None,
            sync=True,
    ) -> Union[RunUDFGenType, RunUDFAGenType]:
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

        roi : numpy.ndarray
            Region of interest as bool mask over the navigation axes of the dataset

        progress : bool
            Show progress bar

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
        Run the `SumUDF` on a data set:

        >>> from libertem.udf.sum import SumUDF
        >>> for result in ctx.run_udf_iter(dataset=dataset, udf=SumUDF()):
        ...     assert np.array(result.buffers[0]["intensity"]).shape == (32, 32)
        >>> np.array(result.buffers[0]["intensity"]).shape
        (32, 32)
        >>> # intensity is the name of the result buffer, defined in the SumUDF
        """
        fn = self._run_async
        if sync:
            fn = self._run_sync

        return fn(
            dataset=dataset,
            udf=udf,
            roi=roi,
            corrections=corrections,
            progress=progress,
            backends=backends,
            plots=plots,
            iterate=True,
        )

    def _run_sync(
            self,
            dataset: DataSet,
            udf: UDF,
            roi: np.ndarray = None,
            corrections: CorrectionSet = None,
            progress: bool = False,
            backends=None,
            plots=None,
            iterate=False,
    ):
        """
        Run the given UDF(s), either returning the final result (when
        :code:`iterate=False` is given), or a generator that yields partial results.
        """
        enable_plotting = bool(plots)

        udf_is_list = isinstance(udf, (tuple, list))
        if not udf_is_list:
            udfs = [udf]
        else:
            udfs = list(udf)

        if enable_plotting:
            plots = self._prepare_plots(udfs, dataset, roi, plots)

        if corrections is None:
            corrections = dataset.get_correction_data()

        if (roi is not None) and (roi.dtype is not np.dtype(bool)):
            warnings.warn(f"ROI dtype is {roi.dtype}, expected bool. Attempting cast to bool.")
            roi = roi.astype(bool)

        def _run_sync_wrap():
            result_iter = UDFRunner(udfs).run_for_dataset_sync(
                dataset=dataset,
                executor=self.executor,
                roi=roi,
                progress=progress,
                corrections=corrections,
                backends=backends,
            )
            for udf_results in result_iter:
                yield udf_results
                if enable_plotting:
                    self._update_plots(
                        plots, udfs, udf_results.buffers, udf_results.damage, force=False
                    )
            if enable_plotting:
                self._update_plots(plots, udfs, udf_results.buffers, udf_results.damage, force=True)

        if iterate:
            return _run_sync_wrap()
        else:
            for udf_results in _run_sync_wrap():
                pass
            if udf_is_list:
                return udf_results.buffers
            else:
                return udf_results.buffers[0]

    def _run_async(
            self,
            dataset: DataSet,
            udf: UDF,
            roi: np.ndarray = None,
            corrections: CorrectionSet = None,
            progress: bool = False,
            backends=None,
            plots=None,
            iterate=False,
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
        udfres_iter = async_generator(sync_generator)

        udf_is_list = isinstance(udf, (tuple, list))

        async def _run_async_wrap():
            async for udf_results in udfres_iter:
                pass
            if udf_is_list:
                return udf_results.buffers
            else:
                return udf_results.buffers[0]

        if iterate:
            return udfres_iter
        else:
            return _run_async_wrap()

    def _get_default_plot_chans(self, buffers):
        from libertem.viz import get_plottable_2D_channels
        return [
            get_plottable_2D_channels(bufferset)
            for bufferset in buffers
        ]

    def _prepare_plots(self, udfs, dataset, roi, plots):
        dry_results = UDFRunner.dry_run(udfs, dataset, roi)

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

    def _update_plots(self, plots, udfs, udf_results, damage, force=False):
        for plot in plots:
            udf = plot.get_udf()
            udf_index = udfs.index(udf)
            plot.new_data(udf_results[udf_index], damage, force=force)

    def display(self, dataset: DataSet, udf: UDF, roi=None):
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

        return _UDFInfo(
            title=udf.__class__.__name__,
            buffers=UDFRunner.inspect_udf(udf, dataset, roi),
        )

    def map(self, dataset: DataSet, f, roi: np.ndarray = None,
            progress: bool = False,
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
        roi : numpy.ndarray
            region of interest as bool mask over the navigation axes of the dataset
        progress : bool
            Show progress bar
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
        results = self.run_udf(
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
