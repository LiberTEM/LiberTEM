from .dm import *
from .raw import RawFileDataSet


class RawFileDataSetFortran(RawFileDataSet):
    def initialize(self, executor):
        super().initialize(executor)
        self.meta.shape._nav_order = 'F'

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield RawPartitionFortran(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def adjust_tileshape(
        self, tileshape: Tuple[int, ...], roi: Optional[np.ndarray]
    ) -> Tuple[int, ...]:
        """
        If C-ordered, return proposed tileshape
        If F-ordered, generates tiles which are close in size to
        the proposed tileshape but tile only in the last signal dimension
        All other tile sig-dims should equal the matching full sig-dim
        Could adjust depth too to get tiles of similar byte-size?

        # NOTE Check how corrections could be broken ??
        """
        sig_shape = self.shape.sig.to_tuple()
        depth, sig_tile = tileshape[0], tileshape[1:]
        if sig_tile == sig_shape:
            # whole frames, nothing to do
            return tileshape
        sig_stub = sig_shape[:-1]
        final_dim = max(1, prod(sig_tile) // prod(sig_stub))
        return (depth,) + sig_stub + (final_dim,)


class FakeDM4Dataset(RawFileDataSet):
    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield DMPartitionFortran(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def adjust_tileshape(
        self, tileshape: Tuple[int, ...], roi: Optional[np.ndarray]
    ) -> Tuple[int, ...]:
        sig_shape = self.shape.sig.to_tuple()
        depth, sig_tile = tileshape[0], tileshape[1:]
        if sig_tile == sig_shape:
            # whole frames, nothing to do
            return tileshape
        sig_stub = sig_shape[1:]
        # try to pick a dimension which gives tiles of similar
        # bytesize to that proposed by the Negotiator
        final_dim = max(1, prod(sig_tile) // prod(sig_stub))
        return (depth,) + (final_dim,) + sig_stub
