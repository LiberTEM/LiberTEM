import numpy as np
import hdbscan

import libertem.analysis.gridmatching as grm
from libertem.utils import make_polar
# import (Match, PointSelection, CorrelationResult,
# make_polar, NotFoundException, make_polar_vectors)


def full_match(centers, zero=None, parameters={}):
    if zero is None:
        zero = centers[0]
    corr = grm.CorrelationResult(
        centers=centers,
        refineds=centers,
        peak_values=np.ones(len(centers)),
        peak_elevations=np.ones(len(centers)),
    )
    return FullMatch.full_match(
        correlation_result=corr,
        zero=zero,
        parameters=parameters
    )


class FullMatch(grm.Match):
    @classmethod
    def full_match(cls, correlation_result: grm.CorrelationResult, zero, cand=[], parameters={}):
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
            min_weight: Minimum weight for a point to be included in the fit
            num_candidates: Maximum number of candidates to return from clustering matching
            min_delta: Minimum length of a potential grid vector
            max_delta: Maximum length of a potential grid vector
        returns:
            (matches: list of Match objects, unmatched: PointCollection, weak: PointCollection)
        '''
        matches = []
        p = cls._make_parameters(parameters)

        filt = correlation_result.peak_elevations >= p['min_weight']

        working_set = grm.PointSelection(correlation_result, selector=filt)

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
            except grm.NotFoundException:
                # print("no match found:\n", working_set)
                new_selector = np.copy(working_set.selector)
                new_selector[zero_selector] = False
                unmatched = working_set.derive(selector=new_selector)
                break
            # We redo the match with optimized parameters
            match = cls._match_all(
                point_selection=working_set, zero=match.zero, a=match.a, b=match.b, parameters=p)
            if len(match) == 0:
                new_selector = np.copy(working_set.selector)
                new_selector[zero_selector] = False
                unmatched = working_set.derive(selector=new_selector)
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
                new_selector[zero_selector] = False
                unmatched = working_set.derive(selector=new_selector)
                break
        weak = grm.PointSelection(correlation_result, selector=np.logical_not(filt))
        return (matches, unmatched, weak)


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


def hdbscan_candidates(points, parameters):
    '''
    Use hdbscan clustering to find potential candidates for lattice vectors.

    We rely on the clusterer and its settings to give us tight and well-populated clusters.
    Then we calculate mean and standard deviation for each cluster
    and then filter again for tightness.
    In the end we return the shortest matches.

    '''
    cutoff = parameters["tolerance"] * parameters["max_delta"]
    clusterer = hdbscan.HDBSCAN(**make_hdbscan_config(points, parameters))
    vectors = grm.make_polar_vectors(points, parameters)
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
        polar_vectors = grm.make_polar_vectors(points, parameters)
        # Brute force is too slow
        # polar_vectors = all_bruteforce(points, min_delta, max_delta, 2)
    return polar_vectors
