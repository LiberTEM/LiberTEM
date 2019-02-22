import numpy as np
import hdbscan


# Calculate coordinates from lattice vectors a, b and indices
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


# Find points that can be generated from the lattice vectors with near integer indices
# Return matched points, corresponding indices, and remainder
def match_all(points, weights, zero, a, b, parameters):
    cutoff = parameters['max_delta'] * parameters["tolerance"]
    try:
        indices = vector_solver(points, zero, a, b)
    except np.linalg.LinAlgError:
        return (np.array([[]]), np.array([[]]), np.array([]), points)
    rounded = np.around(indices)
    errors = np.linalg.norm(np.absolute(indices - rounded), axis=1)
    matched_selector = errors < cutoff
    matched = points[matched_selector]
    matched_indices = rounded[matched_selector].astype(np.int)
    remainder = points[np.invert(matched_selector)]
    return (matched, matched_indices, weights[matched_selector], remainder)


# Try out all combinations of polar_vectors and
# see which one generates most matches
def do_match(points, weights, zero, polar_vectors, parameters):
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

            match = match_all(points, weights, zero, a, b, parameters)
            # At least three points matched
            if len(match[0]) > 2:
                match_matrix[(i, j)] = match
    return match_matrix


# Find the pair of lattice vectors that generates most matches
# Return vectors and additional info about the match
def find_best_vector_match(points, weights, zero, candidates, parameters):
    match_matrix = do_match(points, weights, zero, candidates, parameters)
    if match_matrix:
        # we select the entry with highest number of matches
        candidate_index, match = sorted(
            match_matrix.items(), key=lambda d: len(d[1][1]), reverse=True
        )[0]
        polar_a = candidates[candidate_index[0]]
        polar_b = candidates[candidate_index[1]]
        a, b = make_cartesian(np.array([polar_a, polar_b]))
        return (a, b, match)
    else:
        raise NotFoundException


def optimize(matched, matched_indices, parameters):
    # We stack an index of 1 to the index list for the zero component
    indices = np.hstack([
        np.ones((len(matched_indices), 1)),
        matched_indices
    ])

    (x, residuals, rank, s) = np.linalg.lstsq(indices, matched, rcond=parameters['tolerance'] / 100)
    # (zero, a, b)
    if x.size == 0:
        raise np.linalg.LinAlgError("Optimizing returned empty result")
    return x


def weighted_optimize(matched, matched_indices, weights, parameters):

    # Following
    # https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python

    # We stack an index of 1 to the index list for the zero component
    indices = np.hstack([
        np.ones((len(matched_indices), 1)),
        matched_indices
    ])

    W = np.vstack([weights, weights])

    Aw = indices * np.sqrt(weights[:, np.newaxis])
    Bw = matched * np.sqrt(W.T)
    (x, residuals, rank, s) = np.linalg.lstsq(Aw, Bw, rcond=parameters['tolerance'] / 100)
    # (zero, a, b)
    if x.size == 0:
        raise np.linalg.LinAlgError("Optimizing returned empty result")
    return x


def full_match(points, zero, weights=None, cand=[], parameters={}):
    # FIXME check formatting when included in documentation
    '''
    In the real world, this would distinguish more between finding good candidates
    from sum frames or region of interest
    and then match to individual frames
    The remainder of all frames could be thrown together for an additional round of matching
    to find faint and sparse peaks.

    Parameters
    ----------
    points
        The list of points to match, numpy array of (y, x) coordinate pairs
    zero
        The initial guess for the zero point as numpy array (y, x).
    cand
        List of candidate vectors to use in a first matching round before guessing.
    parameters
        Parameters for the matching.
        min_angle: Minimum angle between two vectors to be considered candidates
        tolerance: Relative position tolerance for peaks to be considered matches
        min_points: Minimum points to try clustering matching. Otherwise match directly
        min_match: Minimum matched clusters from clustering matching to be considered successful
        min_cluster_size_fraction: Tuning parameter for clustering matching. Larger values allow
            smaller or fuzzier clusters.
        min_samples_fraction: Tuning parameter for clustering matching. Larger values allow smaller
            or fuzzier clusters.
        num_candidates: Maximum number of candidates to return from clustering matching
        min_delta: Minimum length of a potential grid vector
        max_delta: Maximum length of a potential grid vector
    '''
    matches = []
    remainder = []
    p = make_params(parameters)

    if weights is None:
        weights = np.ones(len(points))

    filt = weights >= p['min_weight']

    match_points = points[filt]
    match_weights = weights[filt]

    global_remainder = points[np.invert(filt)]

    while True:
        # First, find good candidate
        # Expensive operation, should be done on smaller sample
        # or sum frame result, at least for first passes to match majority
        # of peaks
        if cand:
            polar_candidate_vectors = make_polar(np.array(cand))
            cand = []
        else:
            polar_candidate_vectors = candidates(match_points, p)

        try:
            a, b, (matched, matched_indices, matched_weights, remainder) = find_best_vector_match(
                match_points, match_weights, zero, polar_candidate_vectors, p
            )
            opt_zero, a, b = weighted_optimize(matched, matched_indices, matched_weights, p)
        except NotFoundException:
            # print("no match found:\n", points)
            break

        (matched, matched_indices, matched_weights, remainder) = match_all(
            match_points, match_weights, opt_zero, a, b, p
        )
        if matched.size == 0:
            # print("no endless loop")
            break

        opt_zero, a, b = weighted_optimize(matched, matched_indices, matched_weights, p)

        matches.append((opt_zero, a, b, matched, matched_indices, matched_weights))
        # doesn't span a lattice
        if len(remainder) < 4:
            # print("doesn't span a lattice")
            break

        match_points = remainder
        # We always include the zero point because all lattices have it in
        # common and we filter it out each time
        # Gives additional point to match!
        match_points = np.append(match_points, [zero], axis=0)
    return (matches, np.concatenate((global_remainder, remainder)))


def fastmatch(points, zero, a, b, weights=None, parameters={}):
    # FIXME check formatting when included in documentation
    '''
    This function finds matches for zero point and lattice vectors
    a, b within the list of points.
    This function is much, much faster than the full match.
    It works well to match a large number of point sets
    that share the same lattice vectors, for example from a
    larger grain or monocrystalline material

    Parameters
    ----------
    points
        The list of points to match, numpy array of (y, x) coordinate pairs
    zero
        The near approximate zero point as numpy array (y, x).
    a, b
        The near approximate vectors a, b to match the grid as numpy arrays (y, x).
    weights
        The weight/quality of each point
    parameters
        Parameters for the matching.
        tolerance: Relative position tolerance for peaks to be considered matches
        min_delta: Minimum length of a potential grid vector
        max_delta: Maximum length of a potential grid vector

    returns:
        (zero, a, b, matched, matched_indices, remainder)
    '''
    if weights is None:
        weights = np.ones(len(points))

    p = make_params(parameters)

    filt = weights >= p['min_weight']

    match_points = points[filt]
    match_weights = weights[filt]

    global_remainder = points[np.invert(filt)]

    # We match twice
    try:
        for _ in range(2):
            (matched, matched_indices, matched_weights, remainder) = match_all(
                match_points, match_weights, zero, a, b, p
            )
            if len(matched) >= p['min_match']:
                zero, a, b = weighted_optimize(matched, matched_indices, matched_weights, p)
            else:
                raise np.linalg.LinAlgError("Not enough matched points")
        return (
            zero, a, b, matched, matched_indices, matched_weights,
            np.concatenate((global_remainder, remainder))
        )
    except np.linalg.LinAlgError:
        nan = np.float('nan')
        nan_vec = np.array([nan, nan])
        return (nan_vec, nan_vec, nan_vec, [], [], [], points)


def make_params(p):
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
