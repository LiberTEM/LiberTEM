from libertem.io.utils import get_partition_shape


class DataSetException(Exception):
    pass


class DataSet(object):
    def initialize(self):
        """
        pre-load metadata. this will be executed on a worker node. should return self.
        """
        raise NotImplementedError()

    def get_partitions(self):
        """
        Return a generator over all Partitions in this DataSet
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def shape(self):
        """
        the effective shape, for example imprinted by the scan_size parameter of some dataset impls
        """
        return self.raw_shape

    @property
    def raw_shape(self):
        """
        the "real" shape of the dataset, as it makes sense for the format
        """
        raise NotImplementedError()

    def check_valid(self):
        raise NotImplementedError()

    @classmethod
    def detect_params(cls, path):
        """
        Guess if path can be opened using this DataSet implementation and
        detect parameters.

        returns dict of detected parameters if path matches this dataset type,
        returns False if path is most likely not of a matching type.
        """
        # FIXME: return hints for the user and additional values,
        # for example number of signal elements
        raise NotImplementedError()

    @property
    def diagnostics(self):
        """
        Diagnistics common for all DataSet implementations
        """
        p = next(self.get_partitions())

        return self.get_diagnostics() + [
            {"name": "Partition shape",
             "value": str(p.shape)},

            {"name": "Number of partitions",
             "value": str(len(list(self.get_partitions())))}
        ]

    def get_diagnostics(self):
        """
        Get relevant diagnostics for this dataset, as a list of
        dicts with keys name, value, where value may be string or
        a list of dicts itself. Subclasses should override this method.
        """
        return []

    def partition_shape(self, datashape, framesize, dtype, target_size, min_num_partitions=None):
        """
        Calculate partition shape for the given ``target_size``
        Parameters
        ----------
        datashape : (int, int, int, int)
            size of the whole dataset
        framesize : int
            number of pixels per frame
        dtype : numpy.dtype or str
            data type of the dataset
        target_size : int
            target size in bytes - how large should each partition be?
        min_num_partitions : int
            minimum number of partitions desired, defaults to twice the number of CPU cores
        Returns
        -------
        (int, int, int, int)
            the shape calculated from the given parameters
        """
        return get_partition_shape(datashape, framesize, dtype, target_size,
                                   min_num_partitions)


class Reader(object):
    pass


class DataSetMeta(object):
    def __init__(self, shape, raw_shape, dtype):
        self.shape = shape
        self.raw_shape = raw_shape
        self.dtype = dtype


class Partition(object):
    def __init__(self, meta, partition_slice):
        self.meta = meta
        self.slice = partition_slice

    @property
    def dtype(self):
        return self.meta.dtype

    @property
    def shape(self):
        """
        the shape of the partition; dimensionality depends on format
        """
        return self.slice.shape

    def get_tiles(self, crop_to=None):
        """
        Return a generator over all DataTiles contained in this Partition.

        Note
        ----
        The DataSet may reuse the internal buffer of a tile, so you should
        directly process the tile and not accumulate a number of tiles and then work
        on them.

        Parameters
        ----------

        crop_to : Slice or None
            crop to this slice. datasets may impose additional limits to the shape of the slice
        """
        raise NotImplementedError()

    def get_locations(self):
        # Allow using any worker by default
        return None


class DataTile(object):
    __slots__ = ["data", "tile_slice"]

    def __init__(self, data, tile_slice):
        """
        A unit of data that can easily be processed at once, for example using
        one of the BLAS routines. For large frames, this may be a stack of sub-frame
        tiles.

        Parameters
        ----------
        tile_slice : Slice
            the global coordinates for this data tile

        data : numpy.ndarray
            the data corresponding to the origin/shape of tile_slice
        """
        self.data = data
        self.tile_slice = tile_slice
        assert hasattr(tile_slice.shape, "to_tuple")
        assert data.shape == tuple(tile_slice.shape),\
            "shape mismatch: data=%s, tile_slice=%s" % (data.shape, tile_slice.shape)

    @property
    def flat_data(self):
        """
        Flatten the data.

        The result is a 2D array where each row contains pixel data
        from a single frame. It is just a reshape, so it is a view into
        the original data.
        """
        shape = self.tile_slice.shape
        tileshape = (
            shape.nav.size,    # stackheight, number of frames we process at once
            shape.sig.size,    # framesize, number of pixels per tile
        )
        return self.data.reshape(tileshape)

    def __repr__(self):
        return "<DataTile %r>" % self.tile_slice

    def __getstate__(self):
        return {
            k: getattr(self, k)
            for k in self.__slots__
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
