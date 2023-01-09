import pathlib
import glob
from typing import List
from typing_extensions import Literal

import numpy as np
from skimage.io import imread
import natsort


def load_raw(path, *, shape, dtype):
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_image(path, **kwargs):
    return imread(path, **kwargs)


format_defs = {
    'raw': load_raw,
    'bin': load_raw,
    'npy': np.load,
    'tiff': load_image,
    'tif': load_image,
    'jpg': load_image,
    'png': load_image,
}
# Gotta do this better!
format_T = Literal['raw', 'bin', 'npy', 'tiff', 'tif', 'jpg', 'jpeg', 'png']


sort_methods = {
    'natsorted': natsort.natsorted,
    'os_sorted': natsort.os_sorted,
    'humansorted': natsort.humansorted,
}


sort_types = Literal['natsorted', 'humansorted', 'os_sorted']
sort_enum_names = tuple(en.name for en in natsort.ns)


def join_if_relative(path: pathlib.Path, root_dir: pathlib.Path):
    """
    If path is relative concatenate path with root_dir.
    Otherwise return path in a normalized form.

    # FIXME The behaviour here needs to be adjusted
    as it might be platform dependent. Should probably
    also constrain root_dir to be an absolute path
    during parsing of the inital config file.
    """
    # Must expanduser() before is_absolute()
    filepath = pathlib.Path(path).expanduser()
    if not filepath.is_absolute():
        filepath = root_dir / filepath
    return filepath.expanduser()


def resolve_path_glob(path: pathlib.Path) -> List[pathlib.Path]:
    """
    Resolve path using glob expansion
    A path containing no glob characters
    will still resolve in glob.glob to List[path]
    Return fully resolved paths

    This essentially performs an existence check for files
    Might not be desirable for cases where files are not present
    on the main node but are on other nodes ?
    """
    matches = [pathlib.Path(p).resolve() for p in glob.glob(str(path))]
    if not matches:
        raise FileNotFoundError(f'Found no files matching {path}')
    return matches
