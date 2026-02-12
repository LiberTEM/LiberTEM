from libertem.common import Shape
from libertem.common.math import prod

try:
    import pwd

    def get_owner_name(full_path, stat):
        try:
            name = pwd.getpwuid(stat.st_uid).pw_name
        except KeyError:
            name = stat.st_uid
        return name
# Assume that we are on Windows
# TODO do a proper abstraction layer if this simple solution doesn't work anymore
except ModuleNotFoundError:
    from libertem.common.win_tweaks import get_owner_name  # noqa: F401


def get_partition_shape(
    dataset_shape: Shape,
    target_size_items: int,
    min_num: int,
    num_cores: int
) -> tuple[int, ...]:
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

    num_cores
        Number of cores
    """
    sig_size = dataset_shape.sig.size
    current_p_shape: tuple[int, ...] = ()

    num_cores = max(1, num_cores)
    num_items = dataset_shape.size / target_size_items

    num_per_core = num_items // num_cores + min(1, num_items % num_cores)

    num = max(1, min_num, num_cores * num_per_core)

    target_size_items = int(dataset_shape.size // num)

    for dim in reversed(tuple(dataset_shape.nav)):
        proposed_shape = (dim,) + current_p_shape
        proposed_size = prod(proposed_shape) * sig_size
        if proposed_size <= target_size_items:
            current_p_shape = proposed_shape
        else:
            overshoot = proposed_size / target_size_items
            last_size = max(1, int(dim // overshoot))
            current_p_shape = (last_size,) + current_p_shape
            break

    res = tuple([1] * (len(dataset_shape.nav) - len(current_p_shape))) + current_p_shape
    return res
