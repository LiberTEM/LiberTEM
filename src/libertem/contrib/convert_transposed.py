import os
from typing import Optional, TYPE_CHECKING

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
    def _memmap_flat_shape(self) -> tuple[int, ...]:
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
    """
    Convenience function to convert a transposed Gatan Digital Micrograph
    (.dm4) STEM dataset into a numpy (.npy) file with standard ordering for
    processing with LiberTEM.

    Transposed .dm4 files are stored in :code:`(sig, nav)` order, i.e.
    all frame values for a given signal pixel are stored as blocks,
    which means that extracting a single frame requires traversal of the
    whole file. LiberTEM requires :code:`(nav, sig)` order for processing
    using the UDF interface, i.e. each frame is stored sequentially.

    .. versionadded:: 0.13.0

    Parameters
    ----------

    dm4_path : PathLike
        The path to the .dm4 file
    out_path : PathLike
        The path to the output .npy file
    ctx : libertem.api.Context, optional
        The Context to use to perform the conversion, by default None
        in which case a Dask-based context will be created (optionally)
        following the :code:`num_cpus` argument.
    num_cpus : int, optional
        When :code:`ctx` is not supplied, this argument limits
        the number of CPUs to perform the conversion. This can be
        important as conversion is a RAM-intensive operation and limiting
        the number of CPUs can help reduce bottlenecking.
    dataset_index : int, optional
        If the .dm4 file contains multiple datasets, this can be used
        to select the dataset to convert
        (see :class:`~libertem.io.dataset.dm_single.SingleDMDataSet`)
        for more information.
    progress : bool, optional
        Whether to display a progress bar during conversion, by default False

    Raises
    ------
    DataSetException
        If the DM4 dataset is not stored as transposed
    ValueError
        If both :code:`ctx` and :code:`num_cpus` are supplied
    """
    if ctx is not None and num_cpus is not None:
        raise ValueError('Either supply a Context or number of cpus to use in conversion')
    elif ctx is None:
        ctx = lt.Context.make_with('dask', cpus=num_cpus)
    ds_meta = SingleDMDataSet._read_metadata(dm4_path, use_ds=dataset_index)
    if ds_meta['c_order']:
        raise DataSetException('The DM4 data is not transposed')
    ds = ctx.load('dm', dm4_path, force_c_order=True, dataset_index=dataset_index)
    return _convert_transposed_ds(ctx, ds, out_path, progress=progress)
