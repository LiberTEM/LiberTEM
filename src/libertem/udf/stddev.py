# import functools
import collections

from libertem.common.buffers import BufferWrapper

# from libertem.udf import ResultBuffer, map_frames

VariancePart = collections.namedtuple('VariancePart', ['sum_var', 'sum_im', 'N'])


def my_buffer_batch():
    return {
        'batch': BufferWrapper(
            kind='sig', extra_shape=(3,), dtype='float32'
            )
    }


def my_frame_fn_batch(frame, batch):
    if batch[:, :, 2][0][0] == 0:
        batch[:, :, 0] = 0

    else:
        p0 = VariancePart(sum_var=batch[:, :, 0], sum_im=batch[:, :, 1], N=batch[:, :, 2][0][0])
        p1 = VariancePart(sum_var=0, sum_im=frame, N=1)
        compute_merge = merge(p0, p1)

        sum_var = compute_merge.sum_var

        batch[:, :, 0] = sum_var

    batch[:, :, 1] += frame
    batch[:, :, 2] += 1


def stddev_merge(dest, src):
    sum_var1 = dest['batch'][:, :, 0]
    sum_var2 = src['batch'][:, :, 0]
    sum_im1 = dest['batch'][:, :, 1]
    sum_im2 = src['batch'][:, :, 1]
    obs1 = dest['batch'][:, :, 2][0][0]
    obs2 = src['batch'][:, :, 2][0][0]

    p0 = VariancePart(sum_var=sum_var1, sum_im=sum_im1, N=obs1)
    p1 = VariancePart(sum_var=sum_var2, sum_im=sum_im2, N=obs2)
    compute_merge = merge(p0, p1)

    sum_var, sum_im, N = compute_merge.sum_var, compute_merge.sum_im, compute_merge.N
    dest['batch'][:, :, 0] = sum_var
    dest['batch'][:, :, 1] = sum_im
    dest['batch'][:, :, 2] = N


def merge(p0, p1):
    """
    Given two sets of partitions, with sum of frames
    and sum of variances, compute joint sum of frames
    and sum of variances using one pass algorithm
    """
    if p0.N == 0:
        return VariancePart(sum_var=p1.sum_var, sum_im=p1.sum_im, N=p1.N)
    N = p0.N + p1.N

    # compute mean for each partitions
    mean_A = (p0.sum_im / p0.N)
    mean_B = (p1.sum_im / p1.N)

    # compute mean for joint samples
    delta = mean_B - mean_A
    mean = mean_A + (p1.N * delta) / (p0.N + p1.N)

    # compute sum of images for joint samples
    sum_im_AB = p0.sum_im + p1.sum_im

    # compute sum of variances for joint samples
    delta_P = mean_B - mean
    sum_var_AB = p0.sum_var + p1.sum_var + (p1.N * delta * delta_P)

    return VariancePart(sum_var=sum_var_AB, sum_im=sum_im_AB, N=N)


def run_analysis(ctx, dataset, parameters):
    pass
