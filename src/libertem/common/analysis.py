from io import BytesIO
from typing import (
    Callable, Optional, Union
)

import numpy as np


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
    def __init__(
        self,
        raw_data: np.ndarray,
        visualized: np.ndarray,
        title: str,
        desc: str,
        key: str,
        include_in_download: bool = True,
    ):
        self.include_in_download = include_in_download
        self.raw_data = raw_data
        self._visualized = visualized
        self.title = title
        self.desc = desc
        self.key = key

    def __str__(self):
        result = ""
        for k in ("title", "desc", "key", "raw_data", "visualized"):
            result += f"{k}: {getattr(self, k)}\n"
        return result

    def __repr__(self):
        return "<AnalysisResult: %s>" % self.key

    def __array__(self):
        return np.array(self.raw_data)

    def get_image(self, save_kwargs: Optional[dict] = None) -> BytesIO:
        from libertem.common.viz import encode_image
        return encode_image(self.visualized, save_kwargs=save_kwargs)

    @property
    def visualized(self):
        if callable(self._visualized):
            self._visualized = self._visualized()
        return self._visualized


_ResultsType = Union[list[AnalysisResult], Callable[[], list[AnalysisResult]]]


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
    def __init__(self, results: _ResultsType, raw_results=None):
        self._results = results
        self.raw_results = raw_results

    def __repr__(self):
        return repr(self.results)

    def __getattr__(self, k):
        for result in self.results:
            if result.key == k:
                return result
        raise AttributeError("result with key '{}' not found, have: {}".format(
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
