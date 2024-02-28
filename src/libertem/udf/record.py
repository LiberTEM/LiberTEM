import numpy as np

from libertem.common import Shape
from libertem.udf.base import UDF
from libertem.common.math import prod
from libertem.common.buffers import reshaped_view


class RecordUDF(UDF):
    '''
    Record input data as NumPy .npy file

    Parameters
    ----------

    filename : str or path-like
        Filename where to save. The file will be overwritten if it exists.
    _is_master : bool
        Internal flag, keep at default value.
    '''
    def __init__(self, filename, _is_master=True):
        self._is_master = _is_master
        super().__init__(filename=filename, _is_master=False)

    def get_preferred_input_dtype(self):
        ''
        return self.USE_NATIVE_DTYPE

    @property
    def _ds_shape(self) -> Shape:
        # only valid during run_udf hence _ property
        return self.meta.dataset_shape

    @property
    def _memmap_flat_shape(self) -> tuple[int, ...]:
        # only valid during run_udf hence _ property
        return (prod(self._ds_shape.nav), *self._ds_shape.sig)

    def preprocess(self):
        ''
        if self.meta.roi is not None:
            raise RuntimeError('Recording with ROI is not supported.')
        # create the file once in the preprocess method on the master node
        if self._is_master:
            np.lib.format.open_memmap(
                self.params.filename,
                mode='w+',
                dtype=self.meta.input_dtype,
                shape=tuple(self._ds_shape),
            )

    def get_result_buffers(self):
        ''
        return {}

    def get_task_data(self):
        ''
        m = np.lib.format.open_memmap(
                self.params.filename,
                mode='r+',
                dtype=self.meta.input_dtype,
                shape=tuple(self._ds_shape)
        )
        return {
            'memmap': reshaped_view(m, self._memmap_flat_shape)
        }

    def process_tile(self, tile):
        ''
        self.meta.slice.get(self.task_data.memmap)[:] = tile
