import warnings

import numpy as np
from libertem.utils import calc_coords, within_frame


warnings.warn(
    'gridmatching and fullmatch are moved to '
    'libertem_blobfinder.common in libertem-blobfinder>=0.6 '
    'and are slated for removal in LiberTEM 0.16.',
    DeprecationWarning
)


class CorrelationResult:
    """
    Container class for the result of correlation-based refinement of peak
    positions within a frame.
    """
    def __init__(self, centers, refineds=None, peak_values=None, peak_elevations=None):
        if refineds is None:
            refineds = centers
        if peak_values is None:
            peak_values = np.ones(len(centers))
        if peak_elevations is None:
            peak_elevations = np.ones(len(centers))
        assert all(len(centers) == len(other) for other in [refineds, peak_values, peak_elevations])
        self.centers = centers
        self.refineds = refineds
        self.peak_values = peak_values
        self.peak_elevations = peak_elevations

    def __len__(self):
        return len(self.centers)


class PointSelection:
    '''
    Class that represents a subset of a correlation result.

    Attributes
    ----------
    selector : numpy.ndarray
        Boolean mask for all points in the correlation result, :code:`True` indicating
        selected points.
    '''
    def __init__(self, correlation_result: CorrelationResult, selector=None):
        self.correlation_result = correlation_result
        if selector is None:
            self.selector = np.ones(len(correlation_result.centers), dtype=bool)
        else:
            assert len(correlation_result.centers) == len(selector)
            self.selector = selector

    @property
    def centers(self):
        '''
        numpy.ndarray : Integer centers (y, x) of correlation result masked with :attr:`selector`
        '''
        return self.correlation_result.centers[self.selector]

    @property
    def refineds(self):
        '''
        numpy.ndarray : Refined float centers (y, x) of correlation result masked
                        with :attr:`selector`
        '''
        return self.correlation_result.refineds[self.selector]

    @property
    def peak_values(self):
        '''
        numpy.ndarray : Peak heights of correlation result masked with :attr:`selector`
        '''
        return self.correlation_result.peak_values[self.selector]

    @property
    def peak_elevations(self):
        '''
        numpy.ndarray : Peak elevations of correlation result masked with :attr:`selector`
        '''
        return self.correlation_result.peak_elevations[self.selector]

    def __len__(self):
        return np.sum(self.selector)

    def new_selector(self, selector):
        new_selector = np.copy(self.selector)
        new_selector[self.selector] = selector
        return new_selector

    def derive(self, selector=None):
        if selector is None:
            selector = self.selector
        return PointSelection(self.correlation_result, selector)


class Matcher:
    '''
    The main job of the Matcher object is managing the matching parameters
    and making them available for the various matching routines.

    Parameters
    ----------

    tolerance : float
        Position tolerance in px for peaks to be considered matches
    min_weight : float
        Minimum peak elevation of a peak to be considered for matching
    min_match : int
        Minimum number of matching peaks to be considered a match.
    '''
    def __init__(self, tolerance=3, min_weight=0.1, min_match=3):
        self.tolerance = tolerance
        self.min_match = min_match
        self.min_weight = min_weight

    def fastmatch(self, centers, zero, a, b, refineds=None, peak_values=None, peak_elevations=None):
        '''
        This function creates a Match object from correlation_result and approximates
        for zero point and lattice vectors a and b.
        This function is much, much faster than the full match.
        It works well to match a large number of point sets
        that share the same lattice vectors, for example from a
        larger grain or monocrystalline material. It rejects
        random points or other lattices in the CorrelationResult,
        provided they are not on near-integer positions of zero, a, b.

        Parameters
        ----------

        centers : numpy.ndarray
            numpy.ndarray of shape (n, 2) with integer centers (y, x) of peaks
        refineds : numpy.ndarray
            numpy.ndarray of shape (n, 2) with float centers (y, x) of peaks (subpixel refinement)
        peak_values : numpy.ndarray
            numpy.ndarray of shape (n,) with float maxima of correlation map of peaks
        peak_elevations : numpy.ndarray
            numpy.ndarray of shape (n,) with float elevation of correlation map of peaks.
            See :meth:`libertem_blobfinder.base.correlation.peak_elevation` for details.
        zero : numpy.ndarray
            The near approximate zero point as numpy array (y, x).
        a,b : numpy.ndarray
            The near approximate vectors a, b to match the grid as numpy arrays (y, x).

        Returns
        -------

        Match
            :class:`~libertem.analysis.gridmatching.Match` object with the optimized
            matching result.
        '''
        corr = CorrelationResult(centers, refineds, peak_values, peak_elevations)
        filt = corr.peak_elevations >= self.min_weight

        selection = PointSelection(correlation_result=corr, selector=filt)
        # We match twice because we might catch more peaks in the second run with better parameters
        try:
            match1 = self._match_all(
                point_selection=selection, zero=zero, a=a, b=b
            )
            if len(match1) >= self.min_match:
                match1 = match1.weighted_optimize()
            else:
                raise np.linalg.LinAlgError("Not enough matched points")
            match2 = self._match_all(
                point_selection=selection, zero=match1.zero, a=match1.a, b=match1.b)
            return match2.weighted_optimize()
        except np.linalg.LinAlgError:
            return Match.invalid(corr)

    def affinematch(self, centers, indices, refineds=None, peak_values=None, peak_elevations=None):
        '''
        This function creates a Match object from correlation_result and
        indices for all points. The indices can be non-integer and relative to any
        base vectors zero, a, b, including virtual ones like zero=(0, 0), a=(1, 0), b=(0, 1).

        Refined values for zero, a and b that match the correlated peaks are then derived.

        This match method is very fast, can be robust against a distorted field of view and
        works without determining a lattice. It matches the full CorrelationResult and does
        not reject random points or other outliers.

        It is mathematically equivalent to calculating
        an affine transformation, as inspired by Giulio Guzzinati
        https://arxiv.org/abs/1902.06979

        Parameters
        ----------
        centers : numpy.ndarray
            numpy.ndarray of shape (n, 2) with integer centers (y, x) of peaks
        refineds : numpy.ndarray
            numpy.ndarray of shape (n, 2) with float centers (y, x) of peaks (subpixel refinement)
        peak_values : numpy.ndarray
            numpy.ndarray of shape (n,) with float maxima of correlation map of peaks
        peak_values : numpy.ndarray
            numpy.ndarray of shape (n,) with float elevation of correlation map of peaks.
            See :meth:`libertem_blobfinder.base.correlation.peak_elevation` for details.
        indices : numpy.ndarray
            The indices assigned to each point of the CorrelationResult.

        Returns
        -------
        Match
            :class:`~libertem.analysis.gridmatching.Match`
        '''
        corr = CorrelationResult(centers, refineds, peak_values, peak_elevations)
        match = Match(corr, selector=None, zero=None, a=None, b=None, indices=indices)
        try:
            return match.weighted_optimize()
        except np.linalg.LinAlgError:
            return Match.invalid(corr)

    def _match_all(self, point_selection: PointSelection, zero, a, b):
        '''
        Find points that can be generated from the lattice vectors with near integer indices

        Returns
        -------

        :class:`~libertem.analysis.gridmatching.Match`

        '''
        indices = get_indices(point_selection.refineds, zero, a, b)
        rounded = np.around(indices)
        index_diffs = np.absolute(indices - rounded)
        # We scale the difference from index dimension to the pixel dimension
        diffs = index_diffs * (np.linalg.norm(a), np.linalg.norm(b))
        # We scale far-out differences with the square root of the indices
        # to be more tolerant to small errors of a and b that result in large deviations
        # in absolute position at high indices
        scaled_diffs = diffs / (np.maximum(1, np.abs(indices))**0.5)
        errors = np.linalg.norm(scaled_diffs, axis=1)
        matched_selector = errors < self.tolerance
        matched_indices = rounded[matched_selector].astype(int)
        # remove the ones that weren't matched
        new_selector = point_selection.new_selector(matched_selector)
        result = Match.from_point_selection(
            point_selection, selector=new_selector, zero=zero, a=a, b=b, indices=matched_indices
        )
        return result


class Match(PointSelection):
    '''
    Class that represents a lattice match to a subset of a correlation result

    The attributes are not guaranteed to be correct or sensible for the given lattice.
    The methods :meth:`weighted_optimize` and :meth:`optimize`
    calculate a derived :class:`Match` with a best fit of :attr:`zero`,
    :attr:`a` and :attr:`b` based on the points and the indices.

    Attributes
    ----------

    zero : numpy.ndarray
        Declared zero point (y, x) of the lattice
    a : numpy.ndarray
        Declared "a" vector (y, x) of the lattice
    b : numpy.ndarray
        Declared "b" vector (y, x) of the lattice
    indices : numpy.ndarray
        List of indices (i, j) that are declared to express the matched points as linear combination
        of vectors :code:`a` and :code:`b` with reference to :code:`zero`. The indices
        can be integers or floats, and they can be precise or approximate, depending on the
        matching method.
    '''
    def __init__(self, correlation_result: CorrelationResult,
            selector, zero, a, b, indices):
        self.zero = zero
        self.a = a
        self.b = b
        self.indices = indices
        super().__init__(correlation_result, selector)
        assert len(indices) == len(self)

    def __str__(self):
        result = "zero: %s\n"\
            "a: %s\n"\
            "b: %s"
        return result % (str(self.zero), str(self.a), str(self.b))

    @classmethod
    def from_point_selection(cls, point_selection: PointSelection,
            zero, a, b, indices, selector=None):
        if selector is None:
            selector = point_selection.selector
        return Match(
            correlation_result=point_selection.correlation_result,
            selector=selector, zero=zero, a=a, b=b, indices=indices)

    @classmethod
    def invalid(cls, correlation_result):
        '''
        Match
            A :class:`Match` instance with empty selector and all-'nan' attributes
        '''
        nanvec = np.array([np.nan, np.nan])
        return cls(
            correlation_result=correlation_result,
            selector=np.zeros(len(correlation_result), dtype=bool),
            zero=nanvec,
            a=nanvec,
            b=nanvec,
            indices=np.array([]),
        )

    def isnan(self):
        return np.any(np.isnan(np.array([self.zero, self.a, self.b])))

    @property
    def calculated_refineds(self):
        '''
        numpy.ndarray : Calculated peak positions based on lattice parameters and indices.
        '''
        return calc_coords(self.zero, self.a, self.b, self.indices)

    def calc_coords(self, indices=None, drop_zero=False, frame_shape=None, r=0):
        '''
        Shorthand to calculate peak coordinates.

        Parameters
        ----------

        indices : numpy.ndarray
            Indices to calculate coordinates for. Both an array of (y, x) pairs
            and the output of np.mgrid are supported.
        drop_zero : bool
            Drop the zero order peak. This is important for virtual darkfield imaging.
        frame_shape : Tuple[int, int]
            If set, the peaks are filtered with :meth:`~libertem.analysis.gridmatching.within_frame`
        r : float
            Radius for :meth:`~libertem.analysis.gridmatching.within_frame`

        Returns
        -------

        numpy.ndarray
            A list of (y, x) coordinate pairs for peaks

        Raises
        ------

        ValueError
            If the shape of :code:`indices` is not as expected.
        '''
        if indices is None:
            indices = self.indices
        s = indices.shape
        # Output of mgrid
        if (len(s) == 3) and (s[0] == 2):
            indices = np.concatenate(indices.T)
        # List of (i, j) pairs
        elif (len(s) == 2) and (s[1] == 2):
            pass
        else:
            raise ValueError(
                "Shape of indices is %s, expected (n, 2) or (2, n, m)" % str(indices.shape))

        selector = np.ones(len(indices), dtype=bool)
        if drop_zero:
            nz = np.any(indices != 0, axis=1)
            selector *= nz
        peaks = calc_coords(self.zero, self.a, self.b, indices)
        if frame_shape is not None:
            fy, fx = frame_shape
            selector *= within_frame(peaks, r, fy, fx)
        return peaks[selector]

    @property
    def error(self):
        '''
        float : Weighted average distance between calculated and given peak position.
                numpy.float('inf') if match of length zero.
        '''
        if len(self) > 0:
            diff = np.linalg.norm(self.refineds - self.calculated_refineds, axis=1)
            return (diff * self.peak_elevations).mean() / self.peak_elevations.mean()
        else:
            return np.inf

    def derive(self, selector=None, zero=None, a=None, b=None, indices=None):
        if zero is None:
            zero = self.zero
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if indices is None:
            indices = self.indices
        if selector is None:
            selector = self.selector
        return Match(correlation_result=self.correlation_result, selector=selector,
            zero=zero, a=a, b=b, indices=indices)

    def weighted_optimize(self):
        '''
        Weighted least square optimization of :attr:`zero`, :attr:`a` and :attr:`b`

        Optimization to match the given points and indices using :attr:`peak_elevation` as weight.

        Returns
        -------

        Match
            A new :class:`Match` instance with optimized :attr:`zero`, :attr:`a` and :attr:`b`

        Raises
        ------

        np.linalg.LinAlgError
            If the solver didn't find a solution.
        '''

        # Following
        # https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python

        # We stack an index of 1 to the index list for the zero component
        indices = np.hstack([
            np.ones((len(self.indices), 1)),
            self.indices
        ])

        W = np.vstack([self.peak_elevations, self.peak_elevations])

        Aw = indices * np.sqrt(self.peak_elevations[:, np.newaxis])
        Bw = self.refineds * np.sqrt(W.T)
        (x, residuals, rank, s) = np.linalg.lstsq(
            Aw, Bw, rcond=None
        )
        # (zero, a, b)
        if x.size == 0:
            raise np.linalg.LinAlgError("Optimizing returned empty result")
        zero, a, b = x
        return self.derive(zero=zero, a=a, b=b)

    def optimize(self):
        '''
        Least square optimization of :attr:`zero`, :attr:`a` and :attr:`b`

        Optimization to match the given points and indices.

        Returns
        -------

        Match
            A new :class:`Match` instance with optimized :attr:`zero`, :attr:`a` and :attr:`b`

        Raises
        ------

        np.linalg.LinAlgError
            If the solver didn't find a solution.
        '''
        # We stack an index of 1 to the index list for the zero component
        indices = np.hstack([
            np.ones((len(self.indices), 1)),
            self.indices
        ])

        (x, residuals, rank, s) = np.linalg.lstsq(
            indices, self.refineds, rcond=None)
        # (zero, a, b)
        if x.size == 0:
            raise np.linalg.LinAlgError("Optimizing returned empty result")
        zero, a, b = x
        return self.derive(zero=zero, a=a, b=b)


def get_indices(points, zero, a, b):
    '''
    Find indices to express each point as sum of lattice vectors from zero point

    This could solve for arbitrarily many points, i.e. frame stacks instead of frame by frame
    With that the algorithm could actually match entire frame collections at once.
    '''
    coefficients = np.array((a, b)).T
    target = points - zero
    result = np.linalg.solve(coefficients, target.T).T
    return result


def get_transformation(ref, peaks, center=None, weighs=None):
    '''
    Inspired by Giulio Guzzinati
    https://arxiv.org/abs/1902.06979
    '''
    if center is None:
        center = np.array((0., 0.))

    assert ref.shape == peaks.shape
    A = np.hstack((ref - center, np.ones((len(ref), 1))))
    B = np.hstack((peaks - center, np.ones((len(peaks), 1))))

    if weighs is None:
        pass
    else:
        assert len(ref) == len(weighs)
        W = np.vstack((weighs, weighs, weighs)).T
        A *= W
        B *= W

    (fit, res, rank, s) = np.linalg.lstsq(A, B, rcond=None)
    return fit


def do_transformation(matrix, peaks, center=None):
    if center is None:
        center = np.array((0, 0))
    A = np.hstack((peaks - center, np.ones((len(peaks), 1))))
    B = np.dot(A, matrix)
    return B[:, 0:2] + center


def find_center(matrix):
    target = np.array((0, 0, 1)).T
    diff = np.identity(3)
    diff[2, 2] = 0
    # Find neutral point: solve a*m = a
    result = np.linalg.solve((matrix - diff).T, target)
    return result[0:2]
