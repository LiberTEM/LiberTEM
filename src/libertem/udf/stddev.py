import collections

from libertem.common.buffers import BufferWrapper


VariancePart = collections.namedtuple('VariancePart', ['sum_var', 'sum_im', 'N'])


# def my_buffer_batch():
#     return {
#         'batch': BufferWrapper(
#             kind='sig', extra_shape=(3,), dtype='float32'
#             )
#     }
    

def batch_buffer():
    return {
        'stddev': BufferWrapper(
            kind='sig', dtype='float32'
            ),
        'num_frame': BufferWrapper(
            kind='nav', dtype='float32'
            ),
        'sum_frame': BufferWrapper(
            kind='sig', dtype='float32'
            )
    }


def compute_batch(frame, stddev, sum_frame, num_frame): 
    if num_frame == 0:
        stddev = 0

    else:
        p0 = VariancePart(sum_var=stddev, sum_im=sum_frame, N=num_frame)
        p1 = VariancePart(sum_var=0, sum_im=frame, N=1)
        compute_merge = merge(p0, p1)

        sum_var = compute_merge.sum_var
        stddev = sum_var

    sum_frame += frame
    num_frame += 1  


# def my_frame_fn_batch(frame, batch):
#     if batch[:, :, 2][0][0] == 0:
#         batch[:, :, 0] = 0

#     else:
#         p0 = VariancePart(sum_var=batch[:, :, 0], sum_im=batch[:, :, 1], N=batch[:, :, 2][0][0])
#         p1 = VariancePart(sum_var=0, sum_im=frame, N=1)
#         compute_merge = merge(p0, p1)

#         sum_var = compute_merge.sum_var

#         batch[:, :, 0] = sum_var

#     batch[:, :, 1] += frame
#     batch[:, :, 2] += 1


def batch_merge(dest, src):
    p0 = VariancePart(sum_var=dest['stddev'], sum_im=dest['sum_frame'], N=dest['num_frame'])
    p1 = VariancePart(sum_var=src['stddev'], sum_im=src['sum_frame'], N=src['num_frame'])
    compute_merge = merge(p0, p1)

    dest['stddev'] = compute_merge.sum_var
    dest['sum_frame'] = compute_merge.sum_frame
    dest['num_frame'] = compute_merge.num_frame


# def stddev_merge(dest, src):
#     sum_var1 = dest['batch'][:, :, 0]
#     sum_var2 = src['batch'][:, :, 0]
#     sum_im1 = dest['batch'][:, :, 1]
#     sum_im2 = src['batch'][:, :, 1]
#     obs1 = dest['batch'][:, :, 2][0][0]
#     obs2 = src['batch'][:, :, 2][0][0]

#     p0 = VariancePart(sum_var=sum_var1, sum_im=sum_im1, N=obs1)
#     p1 = VariancePart(sum_var=sum_var2, sum_im=sum_im2, N=obs2)
#     compute_merge = merge(p0, p1)

#     sum_var, sum_im, N = compute_merge.sum_var, compute_merge.sum_im, compute_merge.N
#     dest['batch'][:, :, 0] = sum_var
#     dest['batch'][:, :, 1] = sum_im
#     dest['batch'][:, :, 2] = N


def merge(p0, p1):
    """
    Given two sets of partitions, with sum of frames
    and sum of variances, compute joint sum of frames
    and sum of variances using one pass algorithm

    Parameters 
    ----------
    p0 : collections.namedtuple object 
        Contains information about the first partition, including 
        sum of variances, sum of pixels, and number of frames used

    p1 : collections.namedtuple object
        Contains information about the second partition, including 
        sum of variances, sum of pixels, and number of frames used

    Returns 
    -------
    collections.namedtuple object
        Contains information about the merged partitions, including
        sum of variances, sum of pixels, and number of frames used
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


def run_analysis(ctx, dataset):
    """
    Extended description of the function

    Parameters  
    ----------
    ctx : Context class
    dataset : STEM data (filetype : raw, hd5f)

    Returns
    -------
    bts 
        A buffer that contains sum of variances, sum of pixels, and 
        number of frames used to compute these statistic

    """
    pass_results = ctx.run_udf(
        dataset=dataset,
        fn=compute_batch, 
        make_buffers=batch_buffer,
        merge=batch_merge,
    )

    return (pass_results)
