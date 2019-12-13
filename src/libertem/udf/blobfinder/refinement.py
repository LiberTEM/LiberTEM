import numpy as np
from libertem.utils import frame_peaks

import libertem.analysis.gridmatching as grm

from .patterns import MatchPattern
from .correlation import FastCorrelationUDF, SparseCorrelationUDF, FullFrameCorrelationUDF

# FIXME There's work on flexible FFT backends in scipy
# https://github.com/scipy/scipy/wiki/GSoC-2019-project-ideas#revamp-scipyfftpack
# and discussions about pyfftw performance vs other implementations
# https://github.com/pyFFTW/pyFFTW/issues/264
# For that reason we shoud review the state of Python FFT implementations
# regularly and adapt our choices accordingly
try:
    import pyfftw
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    zeros = pyfftw.zeros_aligned
except ImportError:
    fft = np.fft
    zeros = np.zeros


class RefinementMixin():
    '''
    To be combined with a :class:`libertem.udf.blobfinder.CorrelationUDF`
    using multiple inheritance.

    The mixin must come before the UDF in the inheritance list.

    The subclasses implement a :code:`postprocess` method that calculates a
    refinement of start_zero, start_a and start_b based on the correlation
    result and populates the appropriate result buffers with this refinement
    result.

    This allows combining arbitrary implementations of correlation-based
    matching with arbitrary implementations of the refinement by declaring an
    ad-hoc class that inherits from one subclass of RefinementMixin and one
    subclass of CorrelationUDF.
    '''
    def get_result_buffers(self):
        """
        This adds :code:`zero`, :code:`a`, :code:`b`, :code:`selector`,
        :code:`error` to the superclass result buffer declaration.

        :code:`zero`, :code:`a`, :code:`b`:
            Grid refinement parameters for each frame.
        :code:`selector`:
            Boolean mask of the peaks that were used in the fit.
        :code:`error`:
            Residual of the fit.

        See source code for the exact buffer declaration.
        """
        super_buffers = super().get_result_buffers()
        num_disks = len(self.params.peaks)
        my_buffers = {
            'zero': self.buffer(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'a': self.buffer(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'b': self.buffer(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'selector': self.buffer(
                kind="nav", extra_shape=(num_disks,), dtype="bool"
            ),
            'error': self.buffer(
                kind="nav", dtype="float32",
            ),
        }
        super_buffers.update(my_buffers)
        return super_buffers

    def apply_match(self, index, match):
        """
        Override this method to change how a match is saved in the result
        buffers, for example to support binned processing or ragged result
        arrays.
        """
        r = self.results
        # We cast from float64 to float32 here
        r.zero[index] = match.zero
        r.a[index] = match.a
        r.b[index] = match.b
        r.selector[index] = match.selector
        r.error[index] = match.error


class FastmatchMixin(RefinementMixin):
    '''
    Refinement using :meth:`~libertem.analysis.gridmatching.Matcher.fastmatch`
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        matcher : libertem.analysis.gridmatching.Matcher
            Instance of :class:`~libertem.analysis.gridmatching.Matcher`
        start_zero : numpy.ndarray
            Approximate value (y, x) in px for "zero" point (origin, zero order peak)
        start_a : numpy.ndarray
            Approximate value (y, x) in px for "a" vector.
        start_b : numpy.ndarray
            Approximate value (y, x) in px for "b" vector.
        '''
        super().__init__(*args, **kwargs)

    def postprocess(self):
        super().postprocess()
        p = self.params
        r = self.results
        for index in range(len(self.results.centers)):
            match = p.matcher.fastmatch(
                centers=r.centers[index],
                refineds=r.refineds[index],
                peak_values=r.peak_values[index],
                peak_elevations=r.peak_elevations[index],
                zero=p.start_zero,
                a=p.start_a,
                b=p.start_b,
            )
            self.apply_match(index, match)


class AffineMixin(RefinementMixin):
    '''
    Refinement using :meth:`~libertem.analysis.gridmatching.Matcher.affinematch`
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        matcher : libertem.analysis.gridmatching.Matcher
            Instance of :class:`~libertem.analysis.gridmatching.Matcher`
        indices : numpy.ndarray
            List of indices [(h1, k1), (h2, k2), ...] of all peaks. The indices can be
            non-integer and relative to any base vectors, including virtual ones like
            (1, 0); (0, 1). See documentation of
            :meth:`~libertem.analysis.gridmatching.Matcher.affinematch` for details.
        '''
        super().__init__(*args, **kwargs)

    def postprocess(self):
        super().postprocess()
        p = self.params
        r = self.results
        for index in range(len(self.results.centers)):
            match = p.matcher.affinematch(
                centers=r.centers[index],
                refineds=r.refineds[index],
                peak_values=r.peak_values[index],
                peak_elevations=r.peak_elevations[index],
                indices=p.indices,
            )
            self.apply_match(index, match)


def run_refine(
        ctx, dataset, zero, a, b, match_pattern: MatchPattern, matcher: grm.Matcher,
        correlation='fast', match='fast', indices=None, steps=5, roi=None):
    '''
    Wrapper function to refine the given lattice for each frame by calculating
    approximate peak positions and refining them for each frame using a
    combination of :class:`libertem.udf.blobfinder.CorrelationUDF` and
    :class:`libertem.udf.blobfinder.RefinementMixin`.

    .. versionchanged:: 0.3.0
        Support for :class:`FullFrameCorrelationUDF`
        through parameter :code:`correlation = 'fullframe'`

    Parameters
    ----------

    ctx : libertem.api.Context
        Instance of a LiberTEM :class:`~libertem.api.Context`
    dataset : libertem.io.dataset.base.DataSet
        Instance of a :class:`~libertem.io.dataset.base.DataSet`
    zero : numpy.ndarray
        Approximate value for "zero" point (y, x) in px (origin, zero order
        peak)
    a : numpy.ndarray
        Approximate value for "a" vector (y, x) in px.
    b : numpy.ndarray
        Approximate value for "b" vector (y, x) in px.
    match_pattern : MatchPattern
        Instance of :class:`~MatchPattern`
    matcher : libertem.analysis.gridmatching.Matcher
        Instance of :class:`~libertem.analysis.gridmatching.Matcher` to perform the matching
    correlation : {'fast', 'sparse', 'fullframe'}, optional
        'fast', 'sparse' or 'fullframe' to select :class:`~FastCorrelationUDF`,
        :class:`~SparseCorrelationUDF` or :class:`~FullFrameCorrelationUDF`
    match : {'fast', 'affine'}, optional
        'fast' or 'affine' to select
        :class:`~FastmatchMixin` or :class:`~AffineMixin`
    indices : numpy.ndarray, optional
        Indices to refine. This is trimmed down to
        positions within the frame. As a convenience, for the indices parameter
        this function accepts both shape (n, 2) and (2, n, m) so that
        numpy.mgrid[h:k, i:j] works directly to specify indices. This saves
        boilerplate code when using this function.
        Default: numpy.mgrid[-10:10, -10:10].
    steps : int, optional
        Only for correlation == 'sparse': Correlation steps. See
        :meth:`~SparseCorelationUDF.__init__` for
        details.
    roi : numpy.ndarray, optional
        ROI for :meth:`~libertem.api.Context.run_udf`

    Returns
    -------
    result : Dict[str, BufferWrapper]
        Result buffers of the UDF. See
        :meth:`libertem.udf.blobfinder.correlation.CorrelationUDF.get_result_buffers` and
        :meth:`RefinementMixin.get_result_buffers` for details on the available
        buffers.
    used_indices : numpy.ndarray
        The peak indices that were within the frame.

    Examples
    --------

    >>> dataset = ctx.load(
    ...     filetype="memory",
    ...     data=np.zeros(shape=(2, 2, 128, 128), dtype=np.float32)
    ... )
    >>> (result, used_indices) = run_refine(
    ...     ctx, dataset,
    ...     zero=(64, 64), a=(1, 0), b=(0, 1),
    ...     match_pattern=libertem.udf.blobfinder.RadialGradient(radius=4),
    ...     matcher=grm.Matcher()
    ... )
    >>> result['centers'].data  #doctest: +ELLIPSIS
    array(...)
    '''
    if indices is None:
        indices = np.mgrid[-10:11, -10:11]

    (fy, fx) = tuple(dataset.shape.sig)

    indices, peaks = frame_peaks(
        fy=fy, fx=fx, zero=zero, a=a, b=b,
        r=match_pattern.search, indices=indices
    )
    peaks = peaks.astype('int')

    if correlation == 'fast':
        method = FastCorrelationUDF
    elif correlation == 'sparse':
        method = SparseCorrelationUDF
    elif correlation == 'fullframe':
        method = FullFrameCorrelationUDF
    else:
        raise ValueError(
            "Unknown correlation method %s. Supported are 'fast' and 'sparse'" % correlation
        )

    if match == 'affine':
        mixin = AffineMixin
    elif match == 'fast':
        mixin = FastmatchMixin
    else:
        raise ValueError(
            "Unknown match method %s. Supported are 'fast' and 'affine'" % match
        )

    # The inheritance order matters: FIRST the mixin, which calls
    # the super class methods.
    class MyUDF(mixin, method):
        pass

    udf = MyUDF(
        peaks=peaks,
        indices=indices,
        start_zero=zero,
        start_a=a,
        start_b=b,
        match_pattern=match_pattern,
        matcher=matcher,
        steps=steps
    )

    result = ctx.run_udf(
        dataset=dataset,
        udf=udf,
        roi=roi,
    )
    return (result, indices)
