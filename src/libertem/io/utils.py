import numpy as np
from libertem.common import Shape

try:
    import pwd

    def get_owner_name(full_path, stat):
        return pwd.getpwuid(stat.st_uid).pw_name
# Assume that we are on Windows
# TODO do a proper abstraction layer if this simple solution doesn't work anymore
except ModuleNotFoundError:
    from libertem.win_tweaks import get_owner_name  # noqa: F401


def get_partition_shape(dataset_shape: Shape, target_size_items: int, min_num: int = None):
    """
    Calculate partition shape for the given ``target_size_items``

    Parameters
    ----------

    dataset_shape
        "native" dataset shape

    target_size_items
        target partition size in number of items/pixels

    min_num
        minimum number of partitions
    """
    sig_size = dataset_shape.sig.size
    current_p_shape = ()

    if min_num is None:
        min_num = 1

    target_size_items = min(target_size_items, int(dataset_shape.size // min_num))

    for dim in reversed(dataset_shape.nav):
        proposed_shape = (dim,) + current_p_shape
        proposed_size = np.prod(proposed_shape, dtype=np.int64) * sig_size
        if proposed_size <= target_size_items:
            current_p_shape = proposed_shape
        else:
            overshoot = proposed_size / target_size_items
            last_size = max(1, int(dim // overshoot))
            current_p_shape = (last_size,) + current_p_shape
            break

    res = tuple([1] * (len(dataset_shape.nav) - len(current_p_shape))) + current_p_shape
    return res
