import numpy as np

from libertem.common.buffers import BufferWrapper

'''
Sum up logscaled frames

In comparison to log-scaling the sum, this highlights regions with slightly higher
intensity that appear in may frames in relation to very high intensity in a few frames.

Example:

f1 = (11, 101)
f2 = (11, 1)
f2 = (11, 1)
...
f10 = (11, 1)

log10(sum(f1 ... f10)) == (2.04, 2.04)

sum(log10(f1) ... log10(f10)) == (10.4, 2.04)

'''


def logsum_buffer():
    return {
        'logsum': BufferWrapper(
            kind='sig', dtype='float32'
            ),
    }


def logsum_merge(dest, src):
    dest['logsum'][:] += src['logsum'][:]


def compute_logsum(frame, logsum):
    logsum += np.log(frame - np.min(frame) + 1)


def run_logsum(ctx, dataset):
    return ctx.run_udf(
        dataset=dataset,
        fn=compute_logsum,
        make_buffers=logsum_buffer,
        merge=logsum_merge,
    )
