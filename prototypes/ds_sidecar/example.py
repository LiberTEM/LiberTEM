import tempfile
import numpy as np
import pathlib
import libertem.api as lt
from libertem.io.dataset.raw import RawFileDataSet

from typing import Dict, Union, Any, Callable, Optional
from typing_extensions import Literal

from models import StandardDatasetConfig
from tree import TreeFactory, find_in_tree, does_match
from pydantic import conlist, PositiveInt, ValidationError, BaseModel, validator


class RawDataSetConfig(StandardDatasetConfig, arbitrary_types_allowed=True):
    ds_format: Optional[Literal['raw']] = 'raw'
    nav_shape: conlist(PositiveInt, min_items=1)
    sig_shape: conlist(PositiveInt, min_items=1)
    dtype: np.dtype

    @validator('dtype', pre=True)
    def check_dtype(cls, value):
        if value is not None:
            try:
                value = np.dtype(value)
            except TypeError:
                raise ValueError(f'Cannot cast {value} to dtype')
        return value


class RawFileDataSetConfig(RawFileDataSet):
    """
    Subclass RawFileDataSet to add config file support
    """
    def initialize(self, executor):
        if pathlib.Path(self._path).suffix in ('.toml', '.json'):
            ds_config = get_config(
                self._path,
                RawDataSetConfig,
                pred=dict(
                    config_type='dataset',
                    ds_format='raw'
                )
            )
            self._path = ds_config.path.resolve()
            self._dtype = ds_config.dtype
            self._nav_shape = tuple(ds_config.nav_shape)
            self._sig_shape = tuple(ds_config.sig_shape)
            self._sync_offset = ds_config.sync_offset
        return super().initialize(executor)


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


if __name__ == '__main__':
    nav_shape = (8, 8)
    sig_shape = (16, 16)
    dtype = np.float32
    data: np.ndarray = np.random.uniform(size=nav_shape + sig_shape).astype(dtype)

    with tempfile.TemporaryDirectory() as td:
        tempdir = pathlib.Path(td)
        filepath = tempdir / 'data.raw'
        data.tofile(filepath)

        toml_def = f"""format = 'raw'
path = '{filepath}'
dtype = '{np.dtype(dtype)}'
nav_shape = {list(nav_shape)}
sig_shape = {list(sig_shape)}"""

        toml_path = tempdir / 'config.toml'
        with toml_path.open('w') as fp:
            fp.write(toml_def)

        # Required to provide nav/sig shape for this dataset, would need to change it!
        ds = RawFileDataSetConfig(toml_path, None, nav_shape=(1, 1), sig_shape=(1, 1))
        ctx = lt.Context.make_with('inline')
        ds.initialize(ctx.executor)
