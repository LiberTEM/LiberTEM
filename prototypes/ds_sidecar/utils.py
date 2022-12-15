import pathlib
import glob
from typing import List, Dict, TYPE_CHECKING
from typing_extensions import Literal

import numpy as np
from skimage.io import imread
import natsort

if TYPE_CHECKING:
    from config_base import NestedDict


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


sort_methods = {
    'natsorted': natsort.natsorted,
    'os_sorted': natsort.os_sorted,
    'humansorted': natsort.humansorted,
}


sort_types = Literal['natsorted', 'humansorted', 'os_sorted']
enum_names = tuple(en.name for en in natsort.ns)


def resolve_path_glob(path: pathlib.Path, root_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Resolve path using glob expansion
    If path is relative concatenate path with root_dir.
    A path containing no glob characters
    will still resolve in glob.glob to List[path]
    Return fully resolved paths

    This essentially performs an existence check for files
    Might not be desirable for cases where files are not present
    on the main node but are on other nodes ?
    """
    filepath = pathlib.Path(path).expanduser()
    if not filepath.is_absolute():
        filepath = root_dir / filepath
    return [pathlib.Path(p).resolve() for p in glob.glob(str(filepath))]


class ParserException(Exception):
    ...


class MissingKey:
    ...


def resolve_jsonpath(struct: Dict, jsonpath: str):
    return _resolve_generic(struct, jsonpath, '#/', '/')


def _resolve_generic(struct: Dict, path: str, strip: str, split: str):
    if not isinstance(path, str):
        raise TypeError(f'Cannot resolve key {path}')
    components = path.strip().strip(strip).split(split)
    view = struct
    for c in components:
        if not isinstance(view, dict) or c not in view:
            raise KeyError(f'Cannot resolve key {path}')
        view = view.get(c)
    return view


def find_tree_root(struct: 'NestedDict'):
    if struct.parent is None:
        return struct
    return find_tree_root(struct.parent)


def as_tree(nest, level=0, name=None, do_print=True):
    from config_base import SpecBase

    ident = '  ' * level + f'{nest.__class__.__name__}'
    if name is not None:
        ident = ident + f'   [{name}]'
    lines = [ident]
    if name is not None:
        lines[-1]
    for key, value in nest.items():
        if isinstance(value, SpecBase):
            lines.extend(as_tree(value, level=level + 1, name=key))
    if level == 0:
        tree_string = '\n'.join(lines)
        if do_print:
            print(tree_string)
        else:
            return tree_string
    else:
        return lines
