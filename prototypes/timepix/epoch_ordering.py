import numpy as np
import numba


@numba.njit
def reverse_shift_distance(window: np.ndarray, target: int, threshold: int) -> int:
    """
    Search backward through window until
        abs(window[-idx] - target) < threshold
    and return the reverse index value for this condition
    
    If condition is not met return 0
    """
    shift = 1
    while shift <= window.size:
        value = window[-shift]
        if abs(value - target) < threshold:
            return shift
        shift += 1
    return 0


def combine_slices(*slices):
    """
    Combine elements from slices where the .stop of one
    slice is equal to the .start of the next

    Only valid for step == 1 and increasing slices
    """
    working = None
    for sl in slices:
        assert sl.step in (None, 1)
        assert sl.stop >= sl.start
        if working is None:
            working = sl
            continue
        if sl.start == working.stop:
            working = slice(working.start, sl.stop)
            continue
        else:
            yield working
            working = sl
    if working is not None:
        yield working


def compute_epoch(times: np.ndarray,
                  start: int = 0,
                  threshold: float = 0.1,
                  interval: int = 65536,
                  look_back: int = 1000) -> list[tuple[int, list[slice]]]:
    """
    Assign epoch numbers to an array of generally-sequential
    unsigned ints in the given interval

    Jumps in the sequence of values greater than (interval * threshold)
    are evaluated for membersip of the current, previous or next epoch
    with a look-back distance of look_back. This behaviour handles
    out-of-order values in the sequence both for events arriving early
    and late relative to a change in epoch.

    Returns list of (epoch_number, slices) used to slice from
    times to give all values belonging to the epoch with epoch_number

    TODO Check what happens in really aberrant cases like
    an epoch change at the very start or end of the sequence

    Parameters
    ----------
    times : np.ndarray
        List of unsigned timestamps to categorise into epochs
    start : int, optional
        The epoch number of the first time in the sequence, by default 0
    threshold : float, optional
        The fraction of interval to consider as a potential epoch change,
        by default 0.1. Every time step > (interval * threshold) will
        be evaluated as a potential change of epoch / rollover.
    interval : int, optional
        The full range of timestamp values, by default 65536 (16-bit)
    look_back : int, optional
        The maximum number of values to inspect backwards from a qualifying
        packet to check if it can be assigned to a previous epoch with
        a maximum time difference of (threshold * interval)
    """
    assert (np.issubdtype(times.dtype, np.unsignedinteger)
            or np.issubdtype(times.dtype, np.integer))
    times = times.astype(np.int32)
    dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    # with this definition of dt, the edges are
    # the indexes just after the jump :
    # the value in dt is (a[i] - a[i - 1])
    max_dt = int(interval * threshold)
    edges = np.argwhere(np.abs(dt) > max_dt).squeeze(axis=-1)
    # Add fake edge at end of sequence so we ensure slices go to end
    edges = np.concatenate((edges, [times.size]))
    epochs = [(start, [slice(0, edges[0])])]
    for edge_idx, next_edge in zip(edges[:-1], edges[1:]):
        time_at_event = times[edge_idx]
        window = times[max(0, edge_idx - look_back): edge_idx]
        # shift == n if time_at_event can be sorted left into the window with
        # a jump of less than interval * threshold, else shift == 0
        shift = reverse_shift_distance(window, time_at_event, max_dt)
        if shift == 0:
            # new epoch
            next_epoch = epochs[-1][0] + 1
            epochs.append((next_epoch, [slice(edge_idx, next_edge)]))
        else:
            for _, slices in reversed(epochs):
                in_slices = any(sl.start <= edge_idx - shift < sl.stop for sl in slices)
                if in_slices:
                    slices.append(slice(edge_idx, next_edge))
                    break
    # combine any sequential slices to compact the return value
    return [(epoch, [*combine_slices(*slices)]) for epoch, slices in epochs]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    limit = 65536
    step = 30
    idxs = np.arange(0, 2 * limit, step=step).astype(np.uint16)

    # Add early next epoch pulse
    idxs[65536 // step - 10] = 5
    idxs[65536 // step - 9] = 4
    idxs[65536 // step - 8] = 6
    # Add late previous epoch pulse
    idxs[65536 // step + 10] = 65536 - 5
    idxs[65536 // step + 11] = 65536 - 3
    idxs[65536 // step + 12] = 65536 - 2
    blocks = compute_epoch(idxs)
    for e, b in blocks:
        print(e, b)

    plt.plot(idxs)
    plt.show()