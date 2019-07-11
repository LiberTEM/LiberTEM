import numpy as np

from libertem.utils import make_polar, make_cartesian


def fastmatch(centers, refineds, peak_values, peak_elevations, zero, a, b, parameters):
    corr = CorrelationResult(centers, refineds, peak_values, peak_elevations)
    return Match.fastmatch(corr, zero, a, b, parameters)


def affinematch(centers, refineds, peak_values, peak_elevations, indices):
    corr = CorrelationResult(centers, refineds, peak_values, peak_elevations)
    return Match.affinematch(corr, indices)


class CorrelationResult:
    def __init__(self, centers, refineds, peak_values, peak_elevations):
        assert all(len(centers) == len(other) for other in [refineds, peak_values, peak_elevations])
        self.centers = centers
        self.refineds = refineds
        self.peak_values = peak_values
        self.peak_elevations = peak_elevations

    def __len__(self):
        return len(self.centers)


class PointSelection:
    def __init__(self, correlation_result: CorrelationResult, selector=None):
        self.correlation_result = correlation_result
        if selector is None:
            self.selector = np.ones(len(correlation_result.centers), dtype=np.bool)
        else:
            assert len(correlation_result.centers) == len(selector)
            self.selector = selector

    @property
    def centers(self):
        return self.correlation_result.centers[self.selector]

    @property
    def refineds(self):
        return self.correlation_result.refineds[self.selector]

    @property
    def peak_values(self):
        return self.correlation_result.peak_values[self.selector]

    @property
    def peak_elevations(self):
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


class Match(PointSelection):
    def __init__(self, correlation_result: CorrelationResult,
            selector, zero, a, b, indices, parameters={}):
        self.zero = zero
        self.a = a
        self.b = b
        self.indices = indices
        self.parameters = self._make_parameters(parameters)
        super().__init__(correlation_result, selector)
        assert len(indices) == len(self)

    def __str__(self):
        result = "zero: %s\n"\
            "a: %s\n"\
            "b: %s\n"
        return result % (str(self.zero), str(self.a), str(self.b))

    @classmethod
    def from_point_selection(cls, point_selection: PointSelection,
            zero, a, b, indices, selector=None, parameters={}):
        if selector is None:
            selector = point_selection.selector
        return Match(
            correlation_result=point_selection.correlation_result,
            selector=selector, zero=zero, a=a, b=b, indices=indices, parameters=parameters)

    @property
    def calculated_refineds(self):
        return calc_coords(self.zero, self.a, self.b, self.indices)

    def calc_coords(self, indices=None, drop_zero=False, frame_shape=None, r=0):
        '''
        Shorthand to calculate peak coordinates.

        Parameters
        ----------

        indices:
            Indices to calculate coordinates for. Both an array of (y, x) pairs
            and the output of np.mgrid are supported.
        drop_zero:
            Drop the zero order peak. This is important for virtual darkfield imaging.
        frame_shape : tuple(fy, fx)
            If set, the peaks are filtered with :meth:`~libertem.analysis.gridmatching.within_frame`
        r:
            Radius for :meth:`~libertem.analysis.gridmatching.within_frame`

        Returns
        -------

        A list of (y, x) coordinate paris for peaks
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

        selector = np.ones(len(indices), dtype=np.bool)
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
        diff = np.linalg.norm(self.refineds - self.calculated_refineds, axis=1)
        return (diff * self.peak_elevations).mean() / self.peak_elevations.mean()

    @classmethod
    def _make_parameters(cls, p, a=None, b=None):
        use_default = True
        if a is None:
            upper_a = np.float('inf')
            lower_a = 0
        else:
            use_default = False
            upper_a = np.linalg.norm(a)
            lower_a = upper_a
        if b is None:
            upper_b = np.float('inf')
            lower_b = 0
        else:
            use_default = False
            upper_b = np.linalg.norm(a)
            lower_b = upper_b

        if use_default:
            min_delta = 0
            max_delta = np.float('inf')
        else:
            min_delta = min(upper_a, upper_b)
            max_delta = max(lower_a, lower_b)

        parameters = {
            "min_angle": np.pi / 10,
            "tolerance": 3,
            "min_points": 10,
            "min_match": 3,
            "min_cluster_size_fraction": 4,
            "min_samples_fraction": 20,
            "num_candidates": 7,
            "min_delta": min_delta / 2,
            "max_delta": max_delta * 2,
            "min_weight": 0.1
        }
        parameters.update(p)
        return parameters

    def derive(self, selector=None, zero=None, a=None, b=None, indices=None, parameters={}):
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
        new_parameters = dict(self.parameters)
        new_parameters.update(parameters)
        return Match(correlation_result=self.correlation_result, selector=selector,
            zero=zero, a=a, b=b, indices=indices, parameters=parameters)

    def weighted_optimize(self):

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

    @classmethod
    def invalid(cls, correlation_result):
        nanvec = np.array([np.float('nan'), np.float('nan')])
        return cls(
            correlation_result=correlation_result,
            selector=np.zeros(len(correlation_result), dtype=np.bool),
            zero=nanvec,
            a=nanvec,
            b=nanvec,
            indices=np.array([]),
            parameters={}
        )

    @classmethod
    def fastmatch(cls, correlation_result: CorrelationResult, zero, a, b, parameters={}):
        # FIXME check formatting when included in documentation
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
        correlation_result
            CorrelationResult object with coordinates and weights
        zero
            The near approximate zero point as numpy array (y, x).
        a, b
            The near approximate vectors a, b to match the grid as numpy arrays (y, x).
        parameters
            Parameters for the matching.
            tolerance: Position tolerance in px for peaks to be considered matches
            min_delta: Minimum length of a potential grid vector
            max_delta: Maximum length of a potential grid vector

        returns:
            Match
        '''
        p = cls._make_parameters(parameters, a, b)

        filt = correlation_result.peak_elevations >= p['min_weight']

        selection = PointSelection(correlation_result=correlation_result, selector=filt)
        # We match twice because we might catch more peaks in the second run with better parameters
        try:
            (match1, err) = cls._match_all(
                point_selection=selection, zero=zero, a=a, b=b, parameters=p
            )
            if len(match1) >= p['min_match']:
                match1 = match1.weighted_optimize()
            else:
                raise np.linalg.LinAlgError("Not enough matched points")
            (match2, err) = cls._match_all(
                point_selection=selection, zero=match1.zero, a=match1.a, b=match1.b, parameters=p)
            return match2.weighted_optimize()
        # FIXME proper error handling strategy
        except np.linalg.LinAlgError:
            return cls.invalid(correlation_result)

    @classmethod
    def affinematch(cls, correlation_result: CorrelationResult, indices):
        # FIXME check formatting when included in documentation
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
        correlation_result
            CorrelationResult object with coordinates and weights
        indices
            The indices assigned to each point of the CorrelationResult.

        returns:
            Match
        '''

        match = Match(correlation_result, selector=None, zero=None, a=None, b=None, indices=indices)
        try:
            return match.weighted_optimize()
        # FIXME proper error handling strategy
        except np.linalg.LinAlgError:
            return cls.invalid(correlation_result)

    @classmethod
    def _match_all(cls, point_selection: PointSelection, zero, a, b, parameters):
        '''
        Find points that can be generated from the lattice vectors with near integer indices

        Return match
        '''
        try:
            indices = get_indices(point_selection.refineds, zero, a, b)
        # FIXME proper error handling strategy
        except np.linalg.LinAlgError:
            raise
        rounded = np.around(indices)
        index_diffs = np.absolute(indices - rounded)
        # We scale the difference from index dimension to the pixel dimension
        diffs = index_diffs * (np.linalg.norm(a), np.linalg.norm(b))
        # We scale far-out differences with the square root of the indices
        # to be more tolerant to small errors of a and b that result in large deviations
        # in absolute position at high indices
        scaled_diffs = diffs / (np.maximum(1, np.abs(indices))**0.5)
        errors = np.linalg.norm(scaled_diffs, axis=1)
        matched_selector = errors < parameters["tolerance"]
        matched_indices = rounded[matched_selector].astype(np.int)
        # remove the ones that weren't matched
        new_selector = point_selection.new_selector(matched_selector)
        result = cls.from_point_selection(
            point_selection, selector=new_selector, zero=zero, a=a, b=b, indices=matched_indices
        )
        return (result, np.linalg.norm(errors[matched_selector]))

    @classmethod
    def _do_match(cls, point_selection: PointSelection, zero, polar_vectors, parameters):
        '''
        Return a matrix with matches of all pairwise combinations of polar_vectors
        '''
        match_matrix = {}
        # we test all pairs of candidate vectors
        # and populate match_matrix
        for i in range(len(polar_vectors)):
            for j in range(i + 1, len(polar_vectors)):
                a = polar_vectors[i]
                b = polar_vectors[j]
                # too parallel, not good lattice vectors
                if not angle_check(np.array([a]), np.array([b]), parameters['min_angle']):
                    continue
                if a[0] > b[0]:
                    bb = a
                    aa = b
                    ii = j
                    jj = i
                else:
                    aa = a
                    bb = b
                    ii = i
                    jj = j
                aa, bb = make_cartesian(np.array([aa, bb]))

                (match, error) = cls._match_all(
                    point_selection=point_selection, zero=zero, a=aa, b=bb, parameters=parameters)
                # At least three points matched
                if len(match) > 2:
                    match_matrix[(ii, jj)] = (match, error)
        return match_matrix

    @classmethod
    def _find_best_vector_match(cls, point_selection: PointSelection, zero, candidates, parameters):
        '''
        Return the match that matches with the best figure of merit

        Good properties for vectors are
        * Matching many points in the result
        * Orthogonal
        * Equal length
        * Short

        The function implements a heuristic to calculate a figure of merit that boosts
        candidates for each of the criteria that they fulfill or nearly fulfill.

        FIXME the figure of merit is currently determined heuristically based on discrete
        thresholds. A smooth function would be more elegant.

        FIXME improve this on more real-world examples; define test cases.
        '''
        def fom(d):
            m = d[1][0]
            err = d[1][1]
            na = np.linalg.norm(m.a)
            nb = np.linalg.norm(m.b)
            # Matching many points is good
            res = len(m)**2
            # Low error is good
            if err != 0:
                res /= err**2
            # Boost for orthogonality
            if np.abs(np.dot(m.a, m.b)) < 0.1 * na * nb:
                res *= 5
            elif np.abs(np.dot(m.a, m.b)) < 0.3 * na * nb:
                res *= 2
            # Boost fo nearly equal length
            if np.abs(na - nb) < 0.1 * max(na, nb):
                res *= 5
            if np.abs(na - nb) < 0.3 * max(na, nb):
                res *= 2
            # The division favors short vectors
            res /= na
            res /= nb
            return res

        match_matrix = cls._do_match(point_selection, zero, candidates, parameters)
        if match_matrix:
            # we select the entry with highest figure of merit (fom)
            candidate_index, (match, error) = max(match_matrix.items(), key=fom)
            return match
        else:
            raise NotFoundException


def calc_coords(zero, a, b, indices):
    '''
    Calculate coordinates from lattice vectors a, b and indices
    '''
    coefficients = np.array((a, b))
    return zero + np.dot(indices, coefficients)


def within_frame(peaks, r, fy, fx):
    '''
    Return a boolean vector indicating peaks that are within (r, r) and (fy - r, fx - r)
    '''
    selector = (peaks >= (r, r)) * (peaks < (fy - r, fx - r))
    return selector.all(axis=-1)


def size_filter(polar, min_delta, max_delta):
    '''
    Accept a list of polar vectors
    Return a list of polar vectors with length between min_delta and max_delta
    '''
    select = (polar[:, 0] >= min_delta) * (polar[:, 0] <= max_delta)
    return polar[select]


def angle_check(p1, p2, limit):
    '''
    Check if p1 and p2 have an angle difference of at least limit,
    both parallel or antiparallel
    '''
    diff = np.absolute(p1[:, 1] - p2[:, 1]) % np.pi
    return (diff > limit) * (diff < (np.pi - limit))


def make_polar_vectors(coords, parameters):
    '''
    Calculate all unique pairwise connecting vectors between points in coords.

    The vectors are filtered with parameters["min_delta"] and parameters["max_delta"]
    to avoid calculating for unwanted higher order or random smaller vectors
    '''
    # sort by x coordinate so that we have always positive x difference vectors
    sort_indices = np.argsort(coords[:, 1])
    coords = coords[sort_indices]
    i, j = np.mgrid[0: len(coords), 0: len(coords)]
    selector = j > i
    deltas = coords[j[selector]] - coords[i[selector]]
    polar = make_polar(deltas)
    return size_filter(polar, parameters["min_delta"], parameters["max_delta"])


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


class NotFoundException(Exception):
    pass
