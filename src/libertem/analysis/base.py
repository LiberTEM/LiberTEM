import typing

import numpy as np

from libertem.viz import encode_image, visualize_simple, CMAP_CIRCULAR_DEFAULT


class AnalysisResult:
    """
    This class represents a single 2D image result from an Analysis.

    Instances of this class are contained in an :class:`AnalysisResultSet`.

    Attributes
    ----------
    raw_data : numpy.ndarray
        The raw numerical data of this result
    visualized : numpy.ndarray
        Visualized result as :class:`numpy.ndarray` with RGB or RGBA values
    title : str
        Title for the GUI
    desc : str
        Short description in the GUI
    key : str
        Key to identify the result in an :class:`AnalysisResultSet`
    """
    def __init__(self, raw_data, visualized, title, desc, key, include_in_download=True):
        self.include_in_download = include_in_download
        self.raw_data = raw_data
        self._visualized = visualized
        self.title = title
        self.desc = desc
        self.key = key

    def __str__(self):
        result = ""
        for k in ("title", "desc", "key", "raw_data", "visualized"):
            result += "%s: %s\n" % (k, getattr(self, k))
        return result

    def __repr__(self):
        return "<AnalysisResult: %s>" % self.key

    def __array__(self):
        return np.array(self.raw_data)

    def get_image(self, save_kwargs=None):
        return encode_image(self.visualized, save_kwargs=save_kwargs)

    @property
    def visualized(self):
        if callable(self._visualized):
            self._visualized = self._visualized()
        return self._visualized


class AnalysisResultSet:
    """
    Base class for Analysis result sets. :meth:`libertem.api.Context.run`
    returns an instance of this class or a subclass. Many of the subclasses are
    just introduced to document the Analysis-specific results (keys) of the
    result set and don't introduce new functionality.

    The contents of an :class:`AnalysisResultSet` can be addressed in four different ways:

    1. As a container class with the :attr:`AnalysisResult.key` properties as attributes
       of type :class:`AnalysisResult`.
    2. As a list of :class:`AnalysisResult` objects.
    3. As an iterator of :class:`AnalysisResult` objects (since 0.3).
    4. As a dictionary of :class:`AnalysisResult` objects with the :attr:`AnalysisResult.key`
       properties as keys.

    This allows to implement generic result handling code for an Analysis, for example GUI display,
    as well as specific code for particular Analysis subclasses.

    Attributes
    ----------
    raw_results
        Raw results from the underlying numerical processing, if available, otherwise None.

    Examples
    --------
    >>> mask_shape = tuple(dataset.shape.sig)

    >>> def m0():
    ...    return np.ones(shape=mask_shape)

    >>> def m1():
    ...     result = np.zeros(shape=mask_shape)
    ...     result[0,0] = 1
    ...     return result

    >>> analysis = ctx.create_mask_analysis(
    ...     dataset=dataset, factories=[m0, m1]
    ... )

    >>> result = ctx.run(analysis)

    >>> # As an object with attributes
    >>> print(result.mask_0.title)
    mask 0

    >>> print(result.mask_1.title)
    mask 1

    >>> # As a list
    >>> print(result[0].title)
    mask 0

    >>> # As an iterator
    >>> # New since 0.3.0
    >>> for m in result:
    ...     print(m.title)
    mask 0
    mask 1

    >>> # As a dictionary
    >>> print(result['mask_1'].title)
    mask 1
    """
    def __init__(self, results: typing.List[AnalysisResult], raw_results=None):
        self._results = results
        self.raw_results = raw_results

    def __repr__(self):
        return repr(self.results)

    def __getattr__(self, k):
        for result in self.results:
            if result.key == k:
                return result
        raise AttributeError("result with key '%s' not found, have: %s" % (
            k, ", ".join([r.key for r in self.results])
        ))

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.__getattr__(k)
        return self.results[k]

    def __len__(self):
        return len(self.results)

    def keys(self):
        return [r.key for r in self.results]

    @property
    def results(self):
        if callable(self._results):
            self._results = self._results()
        return self._results

    def __iter__(self):
        return iter(self.results)


class Analysis:
    """
    Abstract base class for Analysis classes.

    An Analysis is the interface between a UDF and the Web API, and handles
    visualization of partial and full results.

    Passing an instance of an :class:`Analysis` sub-class to
    :meth:`libertem.api.Context.run` will generate an :class:`AnalysisResultSet`.
    The content of this result set is governed by the specific implementation of
    the :code:`Analysis` sub-class.

    .. versionadded:: 0.3.0
    """
    # TODO: once we require Py3.8, we can use Literal here:
    # https://www.python.org/dev/peps/pep-0586/
    TYPE: typing.Union[str, None] = None

    def get_results(self, job_results):
        """
        .. deprecated:: 0.4.0
            Use TYPE = 'UDF' and get_udf_results(). See :ref:`job deprecation`.

        Parameters
        ----------
        job_results : list of :class:`~numpy.ndarray`
            raw results from the job

        Returns
        -------
        list of AnalysisResult
            one or more annotated results
        """
        raise NotImplementedError()

    def get_job(self):
        """
        .. deprecated:: 0.4.0
            Use TYPE = 'UDF' and get_udf(). See :ref:`job deprecation`.

        Returns
        -------
        Job
            a Job instance
        """
        raise NotImplementedError()

    def get_udf_results(self, udf_results, roi):
        """
        Convert UDF results to a list of :code:`AnalysisResult`\\ s,
        including visualizations.

        Parameters
        ----------
        udf_results : dics
            raw results from the UDF
        roi : numpy.ndarray or None
            Boolean array of the navigation dimension

        Returns
        -------
        list of AnalysisResult
            one or more annotated results
        """
        raise NotImplementedError()

    def get_udf(self):
        """
        Set TYPE='UDF' on the class and implement this method to run a UDF
        from this analysis
        """
        raise NotImplementedError()

    def get_roi(self):
        """
        Get the region of interest the UDF should be run on. For example,
        the parameters could describe some geometry, which this method should
        convert to a boolean array. See also: :func:`libertem.analysis.getroi.get_roi`

        Returns
        -------
        numpy.ndarray or None
            region of interest for which we want to run our analysis
        """
        raise NotImplementedError()

    def get_complex_results(self, job_result, key_prefix, title, desc):
        raise NotImplementedError()

    def get_parameters(self, parameters):
        """
        Get analysis parameters. Override to set defaults
        """
        raise NotImplementedError()


class BaseAnalysis(Analysis):
    TYPE = 'JOB'

    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.parameters = self.get_parameters(parameters)
        self.parameters.update(parameters)

    def get_roi(self):
        return None

    def get_complex_results(self, job_result, key_prefix, title, desc):
        magn = np.abs(job_result)
        angle = np.angle(job_result)
        wheel = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((job_result.imag, job_result.real))
        return [
            # for compatability, the magnitude has key=key_prefix
            AnalysisResult(
                raw_data=magn,
                visualized=visualize_simple(magn),
                key=key_prefix,
                title="%s [magn]" % title,
                desc="%s [magn]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result.real,
                visualized=visualize_simple(job_result.real),
                key="%s_real" % key_prefix,
                title="%s [real]" % title,
                desc="%s [real]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result.imag,
                visualized=visualize_simple(job_result.imag),
                key="%s_imag" % key_prefix,
                title="%s [imag]" % title,
                desc="%s [imag]" % desc,
            ),
            AnalysisResult(
                raw_data=angle,
                visualized=visualize_simple(angle),
                key="%s_angle" % key_prefix,
                title="%s [angle]" % title,
                desc="%s [angle]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result,
                visualized=wheel,
                key="%s_complex" % key_prefix,
                title="%s [complex]" % title,
                desc="%s [complex]" % desc,
            ),
        ]

    def get_parameters(self, parameters):
        """
        Get analysis parameters. Override to set defaults
        """
        return parameters
