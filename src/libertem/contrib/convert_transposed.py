from typing import Tuple

from libertem.common import Shape
from libertem.udf.record import RecordUDF


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
