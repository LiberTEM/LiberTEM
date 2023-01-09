import pathlib
import glob
from typing import Dict, Union, Any, Callable, List, Optional
from typing_extensions import Literal

import numpy as np
from skimage.io import imread
import natsort

from tree import TreeFactory, find_in_tree, does_match
from pydantic import ValidationError, BaseModel


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


def get_config(
    path: pathlib.Path,
    schema: BaseModel,
    pred: Optional[Union[Dict[str, Any], Callable[[Dict], bool]]] = None,
    strict: bool = False,
):
    """
    Load the config dictionary from file at path and search it
    for configurations which validate against schema (including
    the top level).

    If multiple sub-trees match schema (or strict=True),
    additionally check that the sub-trees match against pred
    if pred is not None. Pred can be a callable to additionally
    validate a sub-tree, or a dictionary of key/value pairs which *must*
    be present in the sub-tree to validate it. This behaviour
    allows us to discriminate against two sub-trees which can both
    be interpreted under schema via casting/defaults, if one is more
    strongly matching than the other.

    # FIXME it should be possible to use the Pydantic model itself to check if
    an attribute of the model came from input data or from the default value

    Raises RuntimeError if either no configs match or more than
    one config matches schema/pred, else return the single valid
    config interpreted using schema.
    """
    nest = TreeFactory.from_file(path)

    def validates(_nest):
        try:
            schema(**_nest.freeze())
            return True
        except ValidationError:
            return False

    compatible = tuple(find_in_tree(nest, validates))
    if pred is not None and (strict or len(compatible) > 1):
        compatible = tuple(v for v in compatible if does_match(v, pred))
    if not compatible:
        raise RuntimeError(f'Unable to find config in {path} '
                           f'compatible with {schema.__class__.__name__}')
    elif len(compatible) > 1:
        raise RuntimeError(f'Multiple compatible configs found in {path}'
                           f'compatible with {schema.__class__.__name__}')

    return schema(**compatible[0].freeze())
