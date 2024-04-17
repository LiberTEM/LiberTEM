import logging
import warnings
warnings.warn(
    'gridmatching and fullmatch are moved to '
    'libertem_blobfinder.common in libertem-blobfinder>=0.6 '
    'and are slated for removal in LiberTEM 0.16.',
    DeprecationWarning
)

import numpy as np  # NOQA: 402
import hdbscan  # NOQA: 402

import libertem.analysis.gridmatching as grm  # NOQA: 402
from libertem.utils import make_polar, make_cartesian  # NOQA: 402


log = logging.getLogger(__name__)


class NotFoundException(Exception):
    pass


class FullMatcher(grm.Matcher):
    '''
    Extension of :class:`~libertem.analysis.gridmatching.Matcher` will full matching

    Include the ability to guess grid parameters from a point cloud. This is separated
    from the other code since it currently only works with :class:`~hdbscan.HDBSCAN`,
    which can be problematic
    to install on some platforms. For that reason it is an optional dependency.

    Parameters
    ----------

    tolerance : float
        Position tolerance in px for peaks to be considered matches
    min_weight : float
        Minimum peak elevation of a peak to be considered for matching
    min_match : int
        Minimum number of matching peaks to be considered a match.
    min_angle : float
        Minimum angle in radians between two vectors to be considered candidates
    min_points : int
        Minimum points to try clustering matching. Otherwise match directly
    min_delta : float
        Minimum length of a potential grid vector
    max_delta : float
        Maximum length of a potential grid vector
    min_candidates : int
        Minimum number of candidates to consider clustering matching successful.
        If not enough are found, the algorithm uses a brute-force search with all
        pairwise vectors between points
    max_candidates : int
        Maximum number of candidates to return from clustering matching
    clusterer
        Instance of sklearn.cluster compatible clusterer. Default is :class:`~hdbscan.HDBSCAN`.
    min_cluster_size_fraction : float
        Tuning parameter for clustering matching with :class:`~hdbscan.HDBSCAN`.
        Larger values allow
        smaller or fuzzier clusters. This is used to adapt the :code:`min_cluster_size`
        parameter of :class:`~hdbscan.HDBSCAN` dynamically to the number of points to be
        matched.
        Set this to :code:`None` to disable dynamic adjustment of :code:`min_cluster_size`.
        If you like to set :code:`min_cluster_size` to a constant value, you can
        set this to :code:`None` and additionally set the :code:`clusterer` parameter with
        your own clusterer object to have direct control over all parameters.
    min_samples_fraction : float
        Tuning parameter for clustering matching with :class:`~hdbscan.HDBSCAN`.
        Larger values allow
        smaller or fuzzier clusters. This is used to adapt the :code:`min_samples`
        parameter of :class:`~hdbscan.HDBSCAN` dynamically to the number of points to be
        matched.
        Set this to :code:`None` to disable dynamic adjustment of :code:`min_samples`.
        If you like to set :code:`min_samples` to a constant value, you can
        set this to :code:`None` and additionally set the :code:`clusterer` parameter with
        your own clusterer object to have direct control over all parameters.
    '''
    def __init__(
            self, tolerance=3, min_weight=0.1, min_match=3, min_angle=np.pi/10,
            min_points=10, min_delta=0, max_delta=np.inf, min_candidates=3,
            max_candidates=7, clusterer=None, min_cluster_size_fraction=4,
            min_samples_fraction=20):

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
            self, centers, zero=None, cand=None,
            refineds=None, peak_values=None, peak_elevations=None):
        '''
        This function extracts a list of Match objects as well two PointSelection objects
        for unmatched and weak points from correlation_result and zero point.
        The zero point is included in each of the matches because it is shared between all grids.

        Parameters
        ----------
        centers : numpy.ndarray
            numpy.ndarray of shape (n, 2) with integer centers (y, x) of peaks. This would typically
            be extracted with :meth:`libertem_blobfinder.common.correlation.get_peaks`
        zero : numpy.ndarray
            Zero point as numpy array (y, x).
        cand : list or numpy.ndarray
            Optional list of candidate vectors (y, x) to use in a first matching round before
            guessing.
        refineds : numpy.ndarray
            numpy.ndarray of shape (n, 2) with float centers (y, x) of peaks (subpixel refinement)
        peak_values : numpy.ndarray
            numpy.ndarray of shape (n,) with float maxima of correlation map of peaks
        peak_elevations : numpy.ndarray
            numpy.ndarray of shape (n,) with float elevation of correlation map of peaks.
            See :meth:`libertem_blobfinder.base.correlation.peak_elevation` for details.

        Returns
        -------
        Tuple[List[libertem.analysis.gridmatching.Match, ...],\
        libertem.analysis.gridmatching.PointSelection,\
        libertem.analysis.gridmatching.PointSelection]
            matches: list of :class:`~libertem.analysis.gridmatching.Match` instances,

            unmatched: instance of :class:`~libertem.analysis.gridmatching.PointSelection`,

            weak: instance of :class:`~libertem.analysis.gridmatching.PointSelection`

        Example
        -------

        >>> peaks = np.array([
        ...     # First peak is zero if not specified otherwise
        ...     # Base lattice vectors (32, 0) and (0, 32)
        ...     (64, 64),
        ...     (32, 32), (32, 64), (32, 96),
        ...     (64, 32), (64, 96),
        ...     (96, 32), (96, 64), (96, 96),
        ... ])
        >>> matcher = FullMatcher()
        >>> (matches, unmatched, weak) = matcher.full_match(peaks)
        >>> m = matches[0]
        >>> assert np.allclose(m.zero, (64, 64))
        >>> assert np.allclose(m.a, (32, 0))
        >>> assert np.allclose(m.b, (0, 32))
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
        ], dtype=bool)

        def listed(working_set, polar_cand):
            return polar_cand

        def guess(working_set, polar_cand):
            return self._candidates(working_set.refineds)

        if cand is not None:
            polar_cand = size_filter(
                make_polar(np.array(cand)),
                min_delta=self.min_delta,
                max_delta=self.max_delta
            )
            candidate_methods = [listed, guess]
        else:
            polar_cand = None
            candidate_methods = [guess]

        while True:
            new_selector = np.copy(working_set.selector)
            # First, find good candidate
            # Expensive operation, should be done on smaller sample
            # or sum frame result, at least for first passes to match majority
            # of peaks
            polar_candidate_vectors = candidate_methods[0](working_set, polar_cand)

            match = self._find_best_vector_match(
                point_selection=working_set, zero=zero,
                candidates=polar_candidate_vectors)
            if match is None:
                candidate_methods = candidate_methods[1:]
                if len(candidate_methods) == 0:
                    break
                else:
                    continue
            matches.append(match)
            # remove the ones that have been matched
            new_selector[match.selector] = False
            if np.count_nonzero(new_selector) >= self.min_match:
                # Add zero point that is shared by all patterns
                new_selector[zero_selector] = True
                working_set = working_set.derive(selector=new_selector)
            else:
                break
        if matches:
            new_selector[zero_selector] = False
        unmatched = working_set.derive(selector=new_selector)
        weak = grm.PointSelection(corr, selector=np.logical_not(filt))
        return (matches, unmatched, weak)

    def make_polar_vectors(self, coords):
        '''
        Calculate all unique pairwise connecting polar vectors between points in coords.

        The pairwise connecting vectors are converted to polar coordinates and
        filtered with parameters :py:attr:`~min_delta` and :py:attr:`~max_delta`
        to avoid calculating for unwanted higher order or random smaller vectors.

        All calculated vectors have a positive or zero x direction.
        '''
        # sort by x coordinate so that we have always positive x difference vectors
        sort_indices = np.argsort(coords[:, 1])
        coords = coords[sort_indices]
        i, j = np.mgrid[0: len(coords), 0: len(coords)]
        selector = j > i
        deltas = coords[j[selector]] - coords[i[selector]]
        polar = make_polar(deltas)
        return size_filter(polar, self.min_delta, self.max_delta)

    def check(self, match):
        if len(match) < self.min_match:
            return False
        papb = make_polar(np.array([match.a, match.b]))
        if len(size_filter(papb, self.min_delta, self.max_delta)) != 2:
            return False
        return angle_check(papb[0:1], papb[1:2], self.min_angle)

    def _tumble(self, point_selection, match):
        if not self.check(match):
            return None
        match = match.weighted_optimize()
        if not self.check(match):
            return None
        match = self._match_all(
            point_selection=point_selection, zero=match.zero, a=match.a, b=match.b)
        if not self.check(match):
            return None
        match = match.weighted_optimize()
        if not self.check(match):
            return None
        else:
            return match

    def _do_match(self, point_selection: grm.PointSelection, zero, polar_vectors):
        '''
        Return a list with matches of all pairwise combinations of polar_vectors
        '''
        match_list = []
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
                else:
                    aa = a
                    bb = b
                aa, bb = make_cartesian(np.array([aa, bb]))
                try:
                    match = self._match_all(
                        point_selection=point_selection, zero=zero, a=aa, b=bb)
                    match = self._tumble(point_selection, match)
                except np.linalg.LinAlgError:
                    continue
                if match is not None:
                    match_list.append(match)
        return match_list

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

        FIXME improve this on more real-world examples; define test cases.

        FIXME The figure of merit function (fom) could be a parameter, implement if need arises.
        '''
        def fom(m):
            na = np.linalg.norm(m.a)
            nb = np.linalg.norm(m.b)
            # Matching many high-quality points is good
            res = np.sum(m.peak_elevations)**2

            # favor orthogonality
            res *= (np.abs(np.cross(m.a, m.b)) / (na * nb))

            # favor equal length
            res *= ((na * nb) / (na**2 + nb**2))

            return res

        match_list = self._do_match(point_selection, zero, candidates)
        if match_list:
            return max(match_list, key=fom)
        else:
            return None

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
        Then we calculate a weighted mean for each cluster.
        In the end we return the shortest matches.
        '''
        # We have special tuning parameters for the default :class:`~hdbscan.HDBSCAN`
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
            if fom > self.tolerance:
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
                    "Matching many points directly, might be computationally intensive: %s" %
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
