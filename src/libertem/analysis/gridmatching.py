import numpy as np
import hdbscan


class CorrelationResult:
    def __init__(self, centers, refineds, peak_values, peak_elevations):
        assert all(len(centers) == len(other) for other in [refineds, peak_values, peak_elevations])
        self.centers = centers
        self.refineds = refineds
        self.peak_values = peak_values
        self.peak_elevations = peak_elevations
        assert all(len(centers) == len(other) for other in [refineds, peak_values, peak_elevations])

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

    @property
    def error(self):
        diff = self.refineds - self.calculated_refineds
        return np.linalg.norm(diff)

    @classmethod
    def _make_parameters(cls, p):
        parameters = {
            "min_angle": np.pi / 5,
            "tolerance": 0.02,
            "min_points": 10,
            "min_match": 3,
            "min_cluster_size_fraction": 4,
            "min_samples_fraction": 20,
            "num_candidates": 7,
            "min_delta": 0,
            "max_delta": np.float('inf'),
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
        (x, residuals, rank, s) = np.linalg.lstsq(Aw, Bw, rcond=self.parameters['tolerance'] / 100)
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
            indices, self.refineds, rcond=self.parameters['tolerance'] / 100)
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
        larger grain or monocrystalline material

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
            tolerance: Relative position tolerance for peaks to be considered matches
            min_delta: Minimum length of a potential grid vector
            max_delta: Maximum length of a potential grid vector

        returns:
            Match
        '''
        p = cls._make_parameters(parameters)

        filt = correlation_result.peak_elevations >= p['min_weight']

        selection = PointSelection(correlation_result=correlation_result, selector=filt)
        # We match twice because we might catch more peaks in the second run with better parameters
        try:
            match1 = cls._match_all(point_selection=selection, zero=zero, a=a, b=b, parameters=p)
            if len(match1) >= p['min_match']:
                match1 = match1.weighted_optimize()
            else:
                raise np.linalg.LinAlgError("Not enough matched points")
            match2 = cls._match_all(
                point_selection=selection, zero=match1.zero, a=match1.a, b=match1.b, parameters=p)
            return match2.weighted_optimize()
        # FIXME proper error handling strategy
        except np.linalg.LinAlgError:
            return cls.invalid(correlation_result)

    @classmethod
    def full_match(cls, correlation_result: CorrelationResult, zero, cand=[], parameters={}):
        # FIXME check formatting when included in documentation
        '''
        This function extracts a list of Match objects as well two PointCollection objects
        for unmatched and weak points from correlation_result and zero point.

        The zero point is included in each of the matches because it is shared between all grids.

        Parameters
        ----------
        correlation_result
            A CorrelationResult object with coordinates and weights
        zero
            Zero point as numpy array (y, x).
        cand
            Optional list of candidate vectors to use in a first matching round before guessing.
        parameters
            Parameters for the matching.
            min_angle: Minimum angle between two vectors to be considered candidates
            tolerance: Relative position tolerance for peaks to be considered matches
            min_points: Minimum points to try clustering matching. Otherwise match directly
            min_match: Minimum matched clusters from clustering matching to be considered successful
            min_cluster_size_fraction: Tuning parameter for clustering matching. Larger values allow
                smaller or fuzzier clusters.
            min_samples_fraction: Tuning parameter for clustering matching. Larger values allow
                smaller or fuzzier clusters.
            num_candidates: Maximum number of candidates to return from clustering matching
            min_delta: Minimum length of a potential grid vector
            max_delta: Maximum length of a potential grid vector

        returns:
            (matches: list of Match objects, unmatched: PointCollection, weak: PointCollection)
        '''
        matches = []
        p = cls._make_parameters(parameters)

        filt = correlation_result.peak_elevations >= p['min_weight']

        working_set = PointSelection(correlation_result, selector=filt)

        zero_selector = np.array([
            np.allclose(correlation_result.centers[i], zero)
            + np.allclose(correlation_result.refineds[i], zero)
            for i in range(len(correlation_result))
        ], dtype=np.bool)

        while True:
            # First, find good candidate
            # Expensive operation, should be done on smaller sample
            # or sum frame result, at least for first passes to match majority
            # of peaks
            if cand:
                polar_candidate_vectors = make_polar(np.array(cand))
                cand = []
            else:
                polar_candidate_vectors = candidates(working_set.refineds, p)

            try:
                match = cls._find_best_vector_match(
                    point_selection=working_set, zero=zero,
                    candidates=polar_candidate_vectors, parameters=p)
                match = match.weighted_optimize()
            except NotFoundException:
                # print("no match found:\n", points)
                break
            # We redo the match with optimized parameters
            match = cls._match_all(
                point_selection=working_set, zero=match.zero, a=match.a, b=match.b, parameters=p)
            if len(match) == 0:
                # print("no endless loop")
                break

            match = match.weighted_optimize()

            matches.append(match)
            new_selector = np.copy(working_set.selector)
            # remove the ones that have been matched
            new_selector[match.selector] = False
            # Test if it spans a lattice
            if sum(new_selector) >= 3:
                # Add zero point that is shared by all patterns
                new_selector[zero_selector] = True
                working_set = working_set.derive(selector=new_selector)
            else:
                # print("doesn't span a lattice")
                unmatched = working_set.derive(selector=new_selector)
                break
        weak = PointSelection(correlation_result, selector=np.logical_not(filt))
        return (matches, unmatched, weak)

    # Find points that can be generated from the lattice vectors with near integer indices
    # Return match
    @classmethod
    def _match_all(cls, point_selection: PointSelection, zero, a, b, parameters):
        cutoff = parameters['max_delta'] * parameters["tolerance"]
        try:
            indices = vector_solver(point_selection.refineds, zero, a, b)
        # FIXME proper error handling strategy
        except np.linalg.LinAlgError:
            raise
        rounded = np.around(indices)
        errors = np.linalg.norm(np.absolute(indices - rounded), axis=1)
        matched_selector = errors < cutoff
        matched_indices = rounded[matched_selector].astype(np.int)
        # remove the ones that weren't matched
        new_selector = point_selection.new_selector(matched_selector)
        return cls.from_point_selection(
            point_selection, selector=new_selector, zero=zero, a=a, b=b, indices=matched_indices)

    # Return a matrix with matches of all pairwise combinations of polar_vectors
    @classmethod
    def _do_match(cls, point_selection: PointSelection, zero, polar_vectors, parameters):
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
                a, b = make_cartesian(np.array([a, b]))

                match = cls._match_all(
                    point_selection=point_selection, zero=zero, a=a, b=b, parameters=parameters)
                # At least three points matched
                if len(match) > 2:
                    match_matrix[(i, j)] = match
        return match_matrix

    # Return the match that matches most points
    @classmethod
    def _find_best_vector_match(cls, point_selection: PointSelection, zero, candidates, parameters):
        match_matrix = cls._do_match(point_selection, zero, candidates, parameters)
        if match_matrix:
            # we select the entry with highest number of matches
            candidate_index, match = sorted(
                match_matrix.items(), key=lambda d: len(d[1]), reverse=True
            )[0]
            return match
        else:
            # FIXME proper error handling
            raise NotFoundException


# Calculate coordinates from lattice vectors a, b and indices
# This might be uselful without a full Match class, so we keep it as function
def calc_coords(zero, a, b, indices):
    coefficients = np.array((a, b))
    return zero + np.dot(indices, coefficients)


# accept list of polar vectors, return list of cartesian vectors
def make_cartesian(polar):
    xes = np.cos(polar[:, 1]) * polar[:, 0]
    yes = np.sin(polar[:, 1]) * polar[:, 0]
    return np.array((yes, xes)).T


# accept list of cartesian vectors, return list of polar vectors
def make_polar(cartesian):
    ds = np.linalg.norm(cartesian, axis=1)
    # (y, x)
    alphas = np.arctan2(cartesian[:, 0], cartesian[:, 1])
    return np.array((ds, alphas)).T


def size_filter(polar, min_delta, max_delta):
    select = (polar[:, 0] >= min_delta) * (polar[:, 0] <= max_delta)
    return polar[select]


# Make sure p1 and p2 have an angle difference of at least limit,
# both parallel or antiparallel
def angle_check(p1, p2, limit):
    diff = np.absolute(p1[:, 1] - p2[:, 1]) % np.pi
    return (diff > limit) * (diff < (np.pi - limit))


# The size filter only retains vectors with absolute values between min_delta
# and max_delta to avoid calculating for unwanted higher order or random smaller vectors
def make_polar_vectors(coords, parameters):
    # sort by x coordinate so that we have always positive x difference vectors
    sort_indices = np.argsort(coords[:, 1])
    coords = coords[sort_indices]
    i, j = np.mgrid[0: len(coords), 0: len(coords)]
    selector = j > i
    deltas = coords[j[selector]] - coords[i[selector]]
    polar = make_polar(deltas)
    return size_filter(polar, parameters["min_delta"], parameters["max_delta"])


def make_hdbscan_config(points, parameters):
    result = {}
    # This is handled here because the defaults depend on the number of points
    defaults = {
        "min_cluster_size": max(len(points) // parameters["min_cluster_size_fraction"], 2),
        "min_samples": max(len(points) // parameters["min_samples_fraction"], 1),
    }
    for (key, default) in defaults.items():
        result[key] = parameters.get(key, default)
    return result


# Use hdbscan clustering to find potential candidates for lattice vectors.
# We rely on the clusterer and its settings to give us tight and well-populated clusters.
# Then we calculate mean and standard deviation for each cluster
# and then filter again for tightness.
# In the end we return the shortest matches.
def hdbscan_candidates(points, parameters):
    cutoff = parameters["tolerance"] * parameters["max_delta"]
    clusterer = hdbscan.HDBSCAN(**make_hdbscan_config(points, parameters))
    vectors = make_polar_vectors(points, parameters)
    clusterer.fit(vectors)
    labels = clusterer.labels_
    cand = []
    for cluster in range(max(labels) + 1):
        selector = labels == cluster
        v = vectors[selector]
        weights = clusterer.probabilities_[selector]
        std = v.std(axis=0)
        mean = np.average(v, axis=0, weights=weights)
        fom = np.linalg.norm(std)
        if fom > cutoff:
            # print("too fuzzy")
            continue

        cand.append(mean)
    # return the shortest candidate vectors
    return np.array(sorted(cand, key=lambda d: d[0])[:parameters['num_candidates']])


def candidates(points, parameters):
    polar_vectors = []
    # Enough "flesh" to cluster
    if len(points) > parameters["min_points"]:
        # Get some candidates
        polar_vectors = hdbscan_candidates(points, parameters)
    # Not enough candidates found, use all pairwise vectors as candidates
    # Tighter vector limits mean less candidates from clustering
    # Adjust as needed because the full match is slow for too many points
    if len(polar_vectors) < parameters["min_match"]:
        if len(points) > parameters["min_points"]:
            print("WARNING matching many points directly: ", len(points))
        polar_vectors = make_polar_vectors(points, parameters)
        # Brute force is too slow
        # polar_vectors = all_bruteforce(points, min_delta, max_delta, 2)
    return polar_vectors


# Find indices to express each point as sum of lattice vectors from zero point
# This could solve for arbitrarily many points, i.e. frame stacks instead of frame by frame
# With that the algorithm could actually match entire frame collections at once
def vector_solver(points, zero, a, b):
    coefficients = np.array((a, b)).T
    target = points - zero
    result = np.linalg.solve(coefficients, target.T).T
    return result


class NotFoundException(Exception):
    pass
