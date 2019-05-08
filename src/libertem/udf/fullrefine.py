import functools

import numba
import numpy as np
import scipy.optimize

import libertem.analysis.gridmatching as grm
from libertem.common.buffers import BufferWrapper


@numba.njit
def correlate_fullframe(params, frame, r0, padding, indices):
    zero = params[0:2]
    a = params[2:4]
    b = params[4:6]
    result = 0
    r02 = r0 ** 2
    (fy, fx) = frame.shape
    bounding = np.ceil(r0 * (1 + padding))

    def radial_gradient(r2):
        if r2 <= r02:
            return np.sqrt(r2) / r0
        # It is critical to provide a smooth ramp at the edge
        # so that the optimizer sees a gradient to converge towards
        # and that small shifts of the position lead to a change of the result
        # Magic number 3 gave best results in practical tests, TBD if optimal in all cases
        elif r2 <= (r0 + 3)**2:
            return 1 + r0/3 - np.sqrt(r2)/3
        else:
            return 0

    for index in range(len(indices)):
        position = zero + indices[index, 0]*a + indices[index, 1]*b
        start = np.maximum(
            np.array((0., 0.)),
            position - np.array((bounding, bounding))
        )
        stop = np.minimum(
            np.array((fy, fx)),
            position + np.array((bounding, bounding)) + np.array((1., 1.))
        )
        for h in range(int(start[0]), int(stop[0])):
            for k in range(int(start[1]), int(stop[1])):
                d2 = (position[0] - h)**2 + (position[1] - k)**2
                result -= radial_gradient(d2) * frame[h, k]

    return result


def get_result_buffers_refine():
    return {
        'intensity': BufferWrapper(
            kind="nav", dtype="float32"
        ),
        'zero': BufferWrapper(
            kind="nav", extra_shape=(2,), dtype="float32"
        ),
        'a': BufferWrapper(
            kind="nav", extra_shape=(2,), dtype="float32"
        ),
        'b': BufferWrapper(
            kind="nav", extra_shape=(2,), dtype="float32"
        ),
    }


def refine(frame, start_zero, start_a, start_b, parameters, indices, intensity, zero, a, b):
    x0 = np.hstack((start_zero, start_a, start_b))
    logframe = np.log(frame - np.min(frame) + 1)
    extra_params = (logframe, parameters['radius'], parameters['padding'], indices)

    res = scipy.optimize.minimize(correlate_fullframe, x0, args=extra_params)

    intensity[:] = res['fun']
    zero[:] = res['x'][0:2]
    a[:] = res['x'][2:4]
    b[:] = res['x'][4:6]


def run_refine(ctx, dataset, zero, a, b, parameters, indices=None):
    '''
    Refine the given lattice for each frame by optimizing the correlation
    with full rendered frames.

    Full frame matching inspired by Christoph Mahr, Knut Müller-Caspary
    and the Bremen group in general

    indices:
        Indices to refine. This is trimmed down to positions within the frame.
        As a convenience, for the indices parameter this function accepts both shape
        (n, 2) and (2, n, m) so that numpy.mgrid[h:k, i:j] works directly to specify indices.
        This saves boilerplate code when using this function. Default: numpy.mgrid[-10:10, -10:10].

    returns:
        (result, used_indices) where result is
        {
            'intensity': BufferWrapper(
                kind="nav", dtype="float32"
            ),
            'zero': BufferWrapper(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'a': BufferWrapper(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'b': BufferWrapper(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
        }
        and used_indices are the indices that were within the frame.
    '''
    if indices is None:
        indices = np.mgrid[-10:10, -10:10]
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

    (fy, fx) = tuple(dataset.shape.sig)

    peaks = grm.calc_coords(zero, a, b, indices).astype('int')

    selector = grm.within_frame(peaks, parameters['radius'], fy, fx)

    indices = indices[selector]

    result = ctx.run_udf(
        dataset=dataset,
        fn=functools.partial(
            refine,
            start_zero=zero,
            start_a=a,
            start_b=b,
            indices=indices,
            parameters=parameters
        ),
        make_buffers=get_result_buffers_refine,
    )
    return (result, indices)
