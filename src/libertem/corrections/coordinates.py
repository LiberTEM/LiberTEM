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


def scale_rotate_flip_y(mat: np.ndarray):
    '''
    Deconstruct a matrix generated with scale() @ rotate() @ flip_y()
    into the individual parameters
    '''
    scale_y = np.linalg.norm(mat[:, 0])
    scale_x = np.linalg.norm(mat[:, 1])
    if not np.allclose(scale_y, scale_x):
        raise ValueError(f'y scale {scale_y} and x scale {scale_x} are different.')

    scan_rot_flip = mat / scale_y
    # 2D cross product
    flip_factor = (
        scan_rot_flip[0, 0] * scan_rot_flip[1, 1]
        - scan_rot_flip[0, 1] * scan_rot_flip[1, 0]
    )
    # Make sure no scale or shear left
    if not np.allclose(np.abs(flip_factor), 1.):
        raise ValueError(
            f'Contains shear: flip factor (2D cross product) is {flip_factor}.'
        )
    flip_y = bool(flip_factor < 0)
    # undo flip_y
    rot = scan_rot_flip.copy()
    rot[:, 0] *= flip_factor

    angle1 = np.arctan2(-rot[1, 0], rot[0, 0])
    angle2 = np.arctan2(rot[0, 1], rot[1, 1])

    # So far not reached in tests since inconsistencies are caught as shear before
    if not np.allclose((np.sin(angle1), np.cos(angle1)), (np.sin(angle2), np.cos(angle2))):
        raise ValueError(
            f'Rotation angle 1 {angle1} and rotation angle 2 {angle2} are inconsistent.'
        )

    return (scale_y, angle1, flip_y)
