from typing import Any, Optional, TYPE_CHECKING

import jsonschema
import numpy as np

from libertem.common import Shape

if TYPE_CHECKING:
    from numpy import typing as nt


class DataSetMeta:
    """
    shape
        "native" dataset shape, can have any dimensionality

    raw_dtype : np.dtype
        dtype used internally in the data set for reading

    dtype : np.dtype
        Best-fitting output dtype. This can be different from raw_dtype, for example
        if there are post-processing steps done as part of reading, which need a different
        dtype. Assumed equal to raw_dtype if not given

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    image_count
        Total number of frames in the dataset

    metadata
        Any metadata offered by the DataSet, not specified yet
    """
    def __init__(
        self,
        shape: Shape,
        image_count: int = 0,
        raw_dtype: "Optional[nt.DTypeLike]" = None,
        dtype: "Optional[nt.DTypeLike]" = None,
        metadata: Optional[Any] = None,
        sync_offset: int = 0
    ):
        self.shape = shape
        if dtype is None:
            dtype = raw_dtype
        self.dtype: np.dtype = np.dtype(dtype)
        self.raw_dtype: np.dtype = np.dtype(raw_dtype)
        self.image_count = image_count
        self.sync_offset = sync_offset
        self.metadata = metadata

    def __getitem__(self, key):
        return self.metadata[key]


class PartitionStructure:
    """
    Structure of the dataset.

    Assumed to be contiguous on the flattened navigation axis.

    Parameters
    ----------

    shape : Shape
        shape of the whole dataset

    dtype : numpy dtype
        The dtype of the data as it is on disk. Can contain endian indicator, for
        example >u2 for big-endian 16bit data.
    """
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/PartitionStructure.schema.json",
        "title": "PartitionStructure",
        "type": "object",
        "properties": {
            "version": {"const": 1},
            "shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
            },
            "sig_dims": {"type": "number"},
            "dtype": {"type": "string"},
        },
        "required": ["version", "shape", "sig_dims", "dtype"]
    }

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def serialize(self):
        data = {
            "version": 1,
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
            shape=shape,
            dtype=np.dtype(data["dtype"]),
        )

    @classmethod
    def from_ds(cls, ds):
        data = {
            "version": 1,
            "shape": list(ds.shape),
            "sig_dims": ds.shape.sig.dims,
            "dtype": str(ds.dtype),
        }
        return cls.from_json(data)
