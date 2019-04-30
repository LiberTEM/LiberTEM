import numpy as np

try:
    import pwd

    def get_owner_name(full_path, stat):
        return pwd.getpwuid(stat.st_uid).pw_name
# Assume that we are on Windows
# TODO do a proper abstraction layer if this simple solution doesn't work anymore
except ModuleNotFoundError:
    from libertem.win_tweaks import get_owner_name  # noqa: F401


def get_partition_shape(datashape, framesize, dtype, target_size, min_num_partitions=None):
    """
    Calculate partition shape for the given ``target_size``
    Parameters
    ----------
    datashape : (int, int, int, int)
        size of the whole dataset
    framesize : int
        number of pixels per frame
    dtype : numpy.dtype or str
        data type of the dataset
    target_size : int
        target size in bytes - how large should each partition be?
    min_num_partitions : int
        minimum number of partitions desired.
    Returns
    -------
    (int, int, int, int)
        the shape calculated from the given parameters
    """
    # FIXME: allow for partitions smaller than one scan row
    # FIXME: allow specifying the "aspect ratio" for a partition?
    num_frames = datashape[0] * datashape[1]
    bytes_per_frame = framesize * np.dtype(str(dtype)).itemsize
    frames_per_partition = target_size // bytes_per_frame
    num_partitions = num_frames // frames_per_partition
    num_partitions = max(min_num_partitions, num_partitions)

    # number of partitions should evenly divide number of scan rows:
    # assert datashape[1] % num_partitions == 0,\
    #     "%d %% %d != 0 (datashape=%r)" % (datashape[1], num_partitions, datashape)

    return (max(1, datashape[0] // num_partitions), datashape[1], datashape[2], datashape[3])
