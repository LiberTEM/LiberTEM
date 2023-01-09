import tempfile
import numpy as np
import pathlib
import libertem.api as lt
from libertem.io.dataset.raw import RawFileDataSet

from typing import Optional
from typing_extensions import Literal

from models import StandardDatasetConfig, DType
from utils import get_config

from pydantic import conlist, PositiveInt, validator


class RawDataSetConfig(StandardDatasetConfig):
    ds_format: Optional[Literal['raw']] = 'raw'
    nav_shape: conlist(PositiveInt, min_items=1)
    sig_shape: conlist(PositiveInt, min_items=1)
    dtype: DType


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
