import os
from typing import Tuple, Optional, TYPE_CHECKING

import libertem.api as lt
from libertem.io.dataset.base import DataSetException
from libertem.io.dataset.dm_single import SingleDMDataSet
from libertem.common import Shape
from libertem.udf.record import RecordUDF

if TYPE_CHECKING:
    from libertem.api import Context, DataSet


class ConvertTransposedDatasetUDF(RecordUDF):
    def get_method(self):
        return self.UDF_METHOD.PARTITION

    @property
    def _ds_shape(self) -> Shape:
        nav_shape = self.meta.dataset_shape.sig.to_tuple()
        sig_shape = self.meta.dataset_shape.nav.to_tuple()
        return Shape(nav_shape + sig_shape, sig_dims=len(sig_shape))

    @property
    def _memmap_flat_shape(self) -> Tuple[int, ...]:
        return (self._ds_shape.nav.size, self._ds_shape.sig.size)

    def process_partition(self, partition):
        # partition will be of shape (n_sig_pix, *ds.shape.nav)
        n_sig_px = partition.shape[0]
        # flatten the nav dimensions
        partition = partition.reshape((n_sig_px, -1))
        # Do the transpose, this is fast but becomes costly
        # the moment we assign into the memmap
        partition = partition.T
        # the LT flat nav origin is actually the sig origin in the memmap
        flat_sig_origin = self.meta.slice.origin[0]
        self.task_data.memmap[
            :, flat_sig_origin:flat_sig_origin + n_sig_px
        ] = partition


def _convert_transposed_ds(
    ctx: 'Context',
    ds: 'DataSet',
    out_path: os.PathLike,
    **run_kwargs,
):
    ctx.run_udf(
        ds,
        ConvertTransposedDatasetUDF(
            out_path,
        ),
        **run_kwargs,
    )


def convert_dm4_transposed(
    dm4_path: os.PathLike,
    out_path: os.PathLike,
    ctx: Optional['Context'] = None,
    num_cpus: Optional[int] = None,
    dataset_index: Optional[int] = None,
    progress: bool = False,
):
    if ctx is not None and num_cpus is not None:
        raise ValueError('Either supply a Context or number of cpus to use in conversion')
    elif ctx is None:
        ctx = lt.Context.make_with('dask', cpus=num_cpus)
    ds_meta = SingleDMDataSet._read_metadata(dm4_path, use_ds=dataset_index)
    if ds_meta['c_order']:
        raise DataSetException('The DM4 data is not transposed')
    ds = ctx.load('dm', dm4_path, force_c_order=True, dataset_index=dataset_index)
    return _convert_transposed_ds(ctx, ds, out_path, progress=progress)
