import numpy as np
import jsonschema

from libertem.common import Shape


class IOCaps:
    """
    I/O capabilities for a dataset (may depend on dataset parameters and concrete format)
    """
    ALL_CAPS = {
        "MMAP",             # .mmap is implemented on the file subclass
        "DIRECT",           # supports direct reading
        "FULL_FRAMES",      # can read full frames
        "SUBFRAME_TILES",   # can read tiles that slice frames into pieces
        "FRAME_CROPS",      # can efficiently crop on signal dimension without needing mmap
    }

    def __init__(self, caps):
        """
        create new capability set
        """
        caps = set(caps)
        for cap in caps:
            self._validate_cap(cap)
        self._caps = caps

    def _validate_cap(self, cap):
        if cap not in self.ALL_CAPS:
            raise ValueError("invalid I/O capability: %s" % cap)

    def __contains__(self, cap):
        return cap in self._caps

    def __getstate__(self):
        return {"caps": self._caps}

    def __setstate__(self, state):
        self._caps = state["caps"]

    def add(self, *caps):
        for cap in caps:
            self._validate_cap(cap)
        self._caps = self._caps.union(caps)

    def remove(self, *caps):
        for cap in caps:
            self._validate_cap(cap)
        self._caps = self._caps.difference(caps)


class DataSetMeta(object):
    def __init__(self, shape: Shape, raw_dtype=None, dtype=None,
                 metadata=None, iocaps: IOCaps = None):
        """
        shape
            "native" dataset shape, can have any dimensionality

        raw_dtype : np.dtype
            dtype used internally in the data set for reading

        dtype : np.dtype
            Best-fitting output dtype. This can be different from raw_dtype, for example
            if there are post-processing steps done as part of reading, which need a different
            dtype. Assumed equal to raw_dtype if not given

        metadata
            Any metadata offered by the DataSet, not specified yet

        iocaps
            I/O capabilities
        """
        self.shape = shape
        if dtype is None:
            dtype = raw_dtype
        self.dtype = np.dtype(dtype)
        self.raw_dtype = np.dtype(raw_dtype)
        self.metadata = metadata
        if iocaps is None:
            iocaps = {}
        self.iocaps = IOCaps(iocaps)

    def __getitem__(self, key):
        return self.metadata[key]


class PartitionStructure:
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/PartitionStructure.schema.json",
        "title": "PartitionStructure",
        "type": "object",
        "properties": {
            "version": {"const": 1},
            "slices": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number", "minItems": 2, "maxItems": 2,
                    }
                },
                "minItems": 1,
            },
            "shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
            },
            "sig_dims": {"type": "number"},
            "dtype": {"type": "string"},
        },
        "required": ["version", "slices", "shape", "sig_dims", "dtype"]
    }

    def __init__(self, shape, slices, dtype):
        """
        Structure of the dataset.

        Assumed to be contiguous on the flattened navigation axis.

        Parameters
        ----------

        slices : List[Tuple[Int]]
            List of tuples [start_idx, end_idx) that partition the data set by the flattened
            navigation axis

        shape : Shape
            shape of the whole dataset

        dtype : numpy dtype
            The dtype of the data as it is on disk. Can contain endian indicator, for
            example >u2 for big-endian 16bit data.
        """
        self.slices = slices
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def serialize(self):
        data = {
            "version": 1,
            "slices": [[s[0], s[1]] for s in self.slices],
            "shape": list(self.shape),
            "sig_dims": self.shape.sig.dims,
            "dtype": str(self.dtype),
        }
        jsonschema.validate(schema=self.SCHEMA, instance=data)
        return data

    @classmethod
    def from_json(cls, data):
        jsonschema.validate(schema=cls.SCHEMA, instance=data)
        shape = Shape(tuple(data["shape"]), sig_dims=data["sig_dims"])
        return PartitionStructure(
            slices=[tuple(item) for item in data["slices"]],
            shape=shape,
            dtype=np.dtype(data["dtype"]),
        )

    @classmethod
    def from_ds(cls, ds):
        data = {
            "version": 1,
            "slices": [
                [p.slice.origin[0], p.slice.origin[0] + p.slice.shape[0]]
                for p in ds.get_partitions()
            ],
            "shape": list(ds.shape),
            "sig_dims": ds.shape.sig.dims,
            "dtype": str(ds.dtype),
        }
        return cls.from_json(data)
