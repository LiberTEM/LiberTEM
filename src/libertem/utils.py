import numpy as np


def make_cartesian(polar):
    '''
    Accept list of polar vectors, return list of cartesian vectors

    polars: ndarray of tuples [(r1, phi1), (r2, phi2), ...]

    returns: ndarray of tuples [(y, x), (y, x), ...]
    '''
    xes = np.cos(polar[..., 1]) * polar[..., 0]
    yes = np.sin(polar[..., 1]) * polar[..., 0]
    return np.array((yes.T, xes.T)).T


def make_polar(cartesian):
    '''
    Accept list of cartesian vectors, return list of polar vectors

    cartesian: ndarray of tuples [(y, x), (y, x), ...]

    returns: ndarray of tuples [(r1, phi1), (r2, phi2), ...]
    '''
    ds = np.linalg.norm(cartesian, axis=-1)
    # (y, x)
    alphas = np.arctan2(cartesian[..., 0], cartesian[..., 1])
    return np.array((ds.T, alphas.T)).T
