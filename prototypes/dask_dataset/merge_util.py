import numpy as np
import math
import dask.array as da


"""
This file contains an implementation of a greedy chunk merging
algorithm which:
 - merges array chunks from right to left dimension-wise
 - merges the smallest pairing of chunks in a dimension
 - breaks ties from the right of a dimension
 - stops when the minimum chunk byte-size is larger than a target
 - (or) stops when there are fewer than a minimum number of chunks

 The entry point is merge_until_target(array, target, min_chunks)
"""


def array_mult(*arrays):
    num_arrays = len(arrays)
    if num_arrays == 1:
        return np.asarray(arrays[0])
    elif num_arrays == 2:
        return np.multiply.outer(*arrays)
    elif num_arrays > 2:
        return np.multiply.outer(arrays[0], array_mult(*arrays[1:]))
    else:
        raise RuntimeError('Unexpected number of arrays')


def get_last_chunked_dim(chunking):
    n_chunks = [len(c) for c in chunking]
    chunked_dims = [idx for idx, el in enumerate(n_chunks) if el > 1]
    try:
        return chunked_dims[-1]
    except IndexError:
        return -1


def get_chunksizes(array, chunking=None):
    if chunking is None:
        chunking = array.chunks
    shape = array.shape
    el_bytes = array.dtype.itemsize
    last_chunked = get_last_chunked_dim(chunking)
    if last_chunked < 0:
        return np.asarray(array.nbytes)
    static_size = math.prod(shape[last_chunked + 1:]) * el_bytes
    chunksizes = array_mult(*chunking[:last_chunked + 1]) * static_size
    return chunksizes


def modify_chunking(chunking, dim, merge_idxs):
    chunk_dim = chunking[dim]
    merge_idxs = tuple(sorted(merge_idxs))
    before = chunk_dim[:merge_idxs[0]]
    after = chunk_dim[merge_idxs[1] + 1:]
    merged_dim = (sum(chunk_dim[merge_idxs[0]:merge_idxs[1] + 1]),)
    new_chunk_dim = tuple(before) + merged_dim + tuple(after)
    chunking = chunking[:dim] + (new_chunk_dim,) + chunking[dim + 1:]
    return chunking


def findall(sequence, val):
    return [idx for idx, e in enumerate(sequence) if e == val]


def neighbour_idxs(sequence, idx):
    max_idx = len(sequence) - 1
    if idx > 0 and idx < max_idx:
        return (idx - 1, idx + 1)
    elif idx == 0:
        return (None, idx + 1)
    elif idx == max_idx:
        return (idx - 1, None)
    else:
        raise


def min_neighbour(sequence, idx):
    left, right = neighbour_idxs(sequence, idx)
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return min([left, right], key=lambda x: sequence[x])


def min_with_min_neighbor(sequence):
    min_val = min(sequence)
    occurences = findall(sequence, min_val)
    min_idx_pairs = [(idx, min_neighbour(sequence, idx)) for idx in occurences]
    pair = [sum(get_values(sequence, idxs)) for idxs in min_idx_pairs]
    min_pair = min(pair)
    min_pair_occurences = findall(pair, min_pair)
    return min_idx_pairs[min_pair_occurences[-1]]  # breaking ties from right


def get_values(sequence, idxs):
    return [sequence[idx] for idx in idxs]


def merge_until_target(array, target, min_chunks):
    chunking = array.chunks
    if array.nbytes < target:
        # A really small dataset, better to treat as one partition
        return tuple([(s,) for s in array.shape])
    chunksizes = get_chunksizes(array)
    while chunksizes.size > min_chunks and chunksizes.min() < target:
        last_chunked_dim = get_last_chunked_dim(chunking)
        if last_chunked_dim < 0:
            # No chunking, by definition complete
            return chunking
        last_chunking = chunking[last_chunked_dim]
        to_merge = min_with_min_neighbor(last_chunking)
        chunking = modify_chunking(chunking, last_chunked_dim, to_merge)
        chunksizes = get_chunksizes(array, chunking=chunking)
    return chunking, chunksizes.min(), chunksizes.max()


if __name__ == '__main__':
    chunking = {0: (8, 4, 3, 1, 7, 6), 1: (2, 7, 3, 7, 6, 1), 2: -1, 3: -1}
    shape = (sum(chunking[0]), sum(chunking[1]), 256, 256)
    sig_dims = 2
    dtype = np.float32
    ar = da.ones(shape, dtype=dtype, chunks=chunking)

    print(f'START: {ar.chunks}')
    target = 128e6
    min_chunks = 3
    chunking, _min, _max = merge_until_target(ar, target, min_chunks)
    print(f'END: {chunking}')
