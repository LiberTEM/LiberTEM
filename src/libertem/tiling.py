class DataTile(object):
    def __init__(self, data, tile_slice):
        """
        Parameters
        ----------
        tile_slice : Slice
            the global coordinates for this data tile

        data : numpy.ndarray
        """
        self.data = data
        self.tile_slice = tile_slice

    def __repr__(self):
        return "<DataTile %r>" % self.tile_slice


class ResultTile(object):
    def __init__(self, data, tile_slice):
        self.data = data
        self.tile_slice = tile_slice

    def __repr__(self):
        return "<ResultTile for slice=%r>" % self.tile_slice

    def copy_to_result(self, result):
        # FIXME: assumes tile size is less than or equal one row of frames. is this true?
        # let's assert it for now:
        assert self.tile_slice.shape[0] == 1

        # (frames, masks) -> (masks, _, frames)
        shape = self.data.shape
        reshaped_data = self.data.reshape(shape[0], 1, shape[1]).transpose()
        result[(Ellipsis,) + self.tile_slice.get()[0:2]] = reshaped_data
        return result
