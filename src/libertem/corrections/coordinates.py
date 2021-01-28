import numpy as np


def scale(factor):
    '''
    .. versionadded:: 0.6.0
    '''
    return np.eye(2) * factor


def rotate(radians):
    '''
    .. versionadded:: 0.6.0
    '''
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return np.array([
        (np.cos(radians), np.sin(radians)),
        (-np.sin(radians), np.cos(radians))
    ])


def rotate_deg(degrees):
    '''
    .. versionadded:: 0.6.0
    '''
    return rotate(np.pi/180*degrees)


def flip_y():
    '''
    .. versionadded:: 0.6.0
    '''
    return np.array([
        (-1, 0),
        (0, 1)
    ])


def flip_x():
    '''
    .. versionadded:: 0.6.0
    '''
    return np.array([
        (1, 0),
        (0, -1)
    ])


def identity():
    '''
    .. versionadded:: 0.6.0
    '''
    return np.eye(2)
