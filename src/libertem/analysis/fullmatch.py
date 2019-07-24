import logging as log

import numpy as np
import hdbscan

import libertem.analysis.gridmatching as grm
from libertem.utils import make_polar, make_cartesian


class NotFoundException(Exception):
    pass


class FullMatcher(grm.Matcher):
    def __init__(
            self, tolerance=3, min_weight=0.1, min_match=3, min_angle=np.pi/10,
            min_points=10, min_delta=0, max_delta=np.float('inf'), min_candidates=3,
            max_candidates=7, clusterer=None, min_cluster_size_fraction=4,
            min_samples_fraction=20):
        '''
        tolerance:
            Position tolerance in px for peaks to be considered matches
        min_weight:
            Minimum peak elevation of a peak to be considered for matching
        min_match:
            Minimum number of matching peaks to be considered a match.
        min_angle:
            Minimum angle between two vectors to be considered candidates
        min_points:
            Minimum points to try clustering matching. Otherwise match directly
        min_delta:
            Minimum length of a potential grid vector
        max_delta:
            Maximum length of a potential grid vector
        min_candidates:
            Minimum number of candidates to consider clustering matching successful.
            If not enough are found, the algorithm uses a brute-force search with all
            pairwise vectors between points
        max_candidates:
            Maximum number of candidates to return from clustering matching
        clusterer:
            sklearn.cluster compatible clusterer. Default is HDBSCAN.
        min_cluster_size_fraction:
            Tuning parameter for clustering matching with HDBSCAN. Larger values allow
            smaller or fuzzier clusters. This is used to adapt the :code:`min_cluster_size`
            parameter of HDBSCAN dynamically to the number of points to be matched.
            Set this to :code:`None` to disable dynamic adjustment of :code:`min_cluster_size`.
            If you like to set :code:`min_cluster_size` to a constant value, you can
            set this to :code:`None`and additionally set the :code:`clusterer` parameter with
            your own clusterer object to have direct control over all parameters.
        min_samples_fraction:
            Tuning parameter for clustering matching with HDBSCAN. Larger values allow
            smaller or fuzzier clusters. This is used to adapt the :code:`min_samples`
            parameter of HDBSCAN dynamically to the number of points to be matched.
            Set this to :code:`None` to disable dynamic adjustment of :code:`min_samples`.
            If you like to set :code:`min_samples` to a constant value, you can
            set this to :code:`None` and additionally set the :code:`clusterer` parameter with
            your own clusterer object to have direct control over all parameters.
        '''
        super().__init__(tolerance=tolerance, min_weight=min_weight, min_match=min_match)
        if clusterer is None:
            clusterer = hdbscan.HDBSCAN()
        self.min_angle = min_angle
        self.min_points = min_points
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.clusterer = clusterer
        self.min_cluster_size_fraction = min_cluster_size_fraction
        self.min_samples_fraction = min_samples_fraction

    def full_match(
            self, centers, zero=None, cand=[],
            refineds=None, peak_values=None, peak_elevations=None):
        # FIXME check formatting when included in documentation
        '''
        This function extracts a list of Match objects as well two PointCollection objects
        for unmatched and weak points from correlation_result and zero point.
        The zero point is included in each of the matches because it is shared between all grids.

        Parameters
        ----------

        centers:
            numpy.ndarray of shape (n, 2) with integer centers (y, x) of peaks
        zero:
            Zero point as numpy array (y, x).
        cand:
            Optional list of candidate vectors (y, x) to use in a first matching round before
            guessing.
        refineds:
            numpy.ndarray of shape (n, 2) with float centers (y, x) of peaks (subpixel refinement)
        peak_values:
            numpy.ndarray of shape (n,) with float maxima of correlation map of peaks
        peak_values:
            numpy.ndarray of shape (n,) with float elevation of correlation map of peaks.
            See :meth:`~libertem.udf.blobfinder.peak_elevation` for details.

        returns:
            (matches: list of Match objects, unmatched: PointCollection, weak: PointCollection)
        '''
        class ExitException(Exception):
            pass

        if zero is None:
            zero = centers[0]

        corr = grm.CorrelationResult(
            centers=centers,
            refineds=refineds,
            peak_values=peak_values,
            peak_elevations=peak_elevations,
        )

        matches = []

        filt = corr.peak_elevations >= self.min_weight

        working_set = grm.PointSelection(corr, selector=filt)

        zero_selector = np.array([
            np.allclose(corr.centers[i], zero)
            + np.allclose(corr.refineds[i], zero)
            for i in range(len(corr))
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
                polar_candidate_vectors = self._candidates(working_set.refineds)

            try:
                match = self._find_best_vector_match(
                    point_selection=working_set, zero=zero,
                    candidates=polar_candidate_vectors)
                match = match.weighted_optimize()
                match = self._match_all(
                    point_selection=working_set, zero=match.zero,
                    a=match.a, b=match.b
                )
                if len(match) == 0:
                    raise ExitException()
                match = match.weighted_optimize()
            except (NotFoundException, np.linalg.LinAlgError, ExitException):
                new_selector = np.copy(working_set.selector)
                new_selector[zero_selector] = False
                unmatched = working_set.derive(selector=new_selector)
                break
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
                new_selector[zero_selector] = False
                unmatched = working_set.derive(selector=new_selector)
                break
        weak = grm.PointSelection(corr, selector=np.logical_not(filt))
        return (matches, unmatched, weak)

    def make_polar_vectors(self, coords):
        '''
        Calculate all unique pairwise connecting vectors between points in coords.

        The vectors are filtered with parameters :py:attr:`~min_delta` and :py:attr:`~max_delta`
        to avoid calculating for unwanted higher order or random smaller vectors
        '''
        # sort by x coordinate so that we have always positive x difference vectors
        sort_indices = np.argsort(coords[:, 1])
        coords = coords[sort_indices]
        i, j = np.mgrid[0: len(coords), 0: len(coords)]
        selector = j > i
        deltas = coords[j[selector]] - coords[i[selector]]
        polar = make_polar(deltas)
        return size_filter(polar, self.min_delta, self.max_delta)

    def _do_match(self, point_selection: grm.PointSelection, zero, polar_vectors):
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
                if not angle_check(np.array([a]), np.array([b]), self.min_angle):
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

                match = self._match_all(
                    point_selection=point_selection, zero=zero, a=aa, b=bb)
                # At least three points matched
                if len(match) > 2:
                    match_matrix[(ii, jj)] = match
        return match_matrix

    def _find_best_vector_match(self, point_selection: grm.PointSelection, zero, candidates):
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
            m = d[1]
            na = np.linalg.norm(m.a)
            nb = np.linalg.norm(m.b)
            # Matching many high-quality points is good
            res = np.sum(m.peak_elevations)**2
            # Boost for orthogonality
            if np.abs(np.dot(m.a, m.b)) < 0.1 * na * nb:
                res *= 5
            elif np.abs(np.dot(m.a, m.b)) < 0.3 * na * nb:
                res *= 2
            # Boost fo nearly equal length
            if np.abs(na - nb) < 0.1 * max(na, nb):
                res *= 2
            # The division favors short vectors
            res /= na
            res /= nb
            return res

        match_matrix = self._do_match(point_selection, zero, candidates)
        if match_matrix:
            # we select the entry with highest figure of merit (fom)
            candidate_index, match = max(match_matrix.items(), key=fom)
            return match
        else:
            raise NotFoundException

    def _make_hdbscan_config(self, points):
        # This is handled here because the defaults depend on the number of points
        defaults = {}
        if self.min_cluster_size_fraction is not None:
            defaults['min_cluster_size'] = max(len(points) // self.min_cluster_size_fraction, 2)
        if self.min_samples_fraction is not None:
            defaults['min_samples'] = max(len(points) // self.min_samples_fraction, 1)
        return defaults

    def _hdbscan_candidates(self, points):
        '''
        Use hdbscan clustering to find potential candidates for lattice vectors.

        We rely on the clusterer and its settings to give us tight and well-populated clusters.
        Then we calculate mean and standard deviation for each cluster
        and then filter again for tightness.
        In the end we return the shortest matches.

        '''
        cutoff = self.tolerance
        # We have special tuning parameters for the default HDBSCAN
        if isinstance(self.clusterer, hdbscan.HDBSCAN):
            defaults = self._make_hdbscan_config(points)
            for key, value in defaults.items():
                setattr(self.clusterer, key, value)
        vectors = self.make_polar_vectors(points)
        self.clusterer.fit(vectors)
        labels = self.clusterer.labels_
        cand = []
        for cluster in range(max(labels) + 1):
            selector = labels == cluster
            v = vectors[selector]
            weights = self.clusterer.probabilities_[selector]
            std = v.std(axis=0)
            mean = np.average(v, axis=0, weights=weights)
            fom = np.linalg.norm(std)
            if fom > cutoff:
                # print("too fuzzy")
                continue

            cand.append(mean)
        # return the shortest candidate vectors
        return np.array(sorted(cand, key=lambda d: d[0])[:self.max_candidates])

    def _candidates(self, points):
        polar_vectors = []
        # Enough "flesh" to cluster
        if len(points) > self.min_points:
            # Get some candidates
            polar_vectors = self._hdbscan_candidates(points)
        # Not enough candidates found, use all pairwise vectors as candidates
        # Tighter vector limits mean less candidates from clustering
        # Adjust as needed because the full match is slow for too many points
        if len(polar_vectors) < self.min_candidates:
            if len(points) > self.min_points:
                log.warn(
                    "Matching many points directly, might be compuationally intensive: %s" %
                    len(points)
                )
            polar_vectors = self.make_polar_vectors(points)
        return polar_vectors


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
