import os
import typing
import logging
import warnings

from ncempy.io.dm import fileDM
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape
from libertem.io.dataset.base.file import OffsetsSizes
from libertem.common.messageconverter import MessageConverter
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta, File,
    IOBackend,
)

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from numpy import typing as nt


class SingleDMDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/DMDatasetParams.schema.json",
        "title": "DMDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "DM"},
            "path": {"type": "string"},
            "nav_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sig_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sync_offset": {"type": "number"},
            "io_backend": {
                "enum": IOBackend.get_supported(),
            },
            "force_c_order": {"type": "boolean"},
        },
        "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        for k in ["nav_shape", "sig_shape", "sync_offset", "force_c_order"]:
            if k in raw_data:
                data[k] = raw_data[k]
        return data


class StackedDMDatasetParams(MessageConverter):
    SCHEMA: dict = {}

    def convert_from_python(self, raw_data):
        return super().convert_from_python(raw_data)

    def convert_to_python(self, raw_data):
        return super().convert_to_python(raw_data)


def _get_metadata(path):
    fh = fileDM(path, on_memory=True)
    if fh.numObjects == 1:
        idx = 0
    else:
        idx = 1
    return {
        'offset': fh.dataOffset[idx],
        'zsize': fh.zSize[idx],
    }


class StackedDMFile(File):
    def get_array_from_memview(self, mem: memoryview, slicing: OffsetsSizes):
        mem = mem[slicing.file_offset:-slicing.skip_end]
        res = np.frombuffer(mem, dtype="uint8")
        itemsize = np.dtype(self._native_dtype).itemsize
        sigsize = int(prod(self._sig_shape))
        cutoff = 0
        cutoff += (
            self.num_frames * itemsize * sigsize
        )
        res = res[:cutoff]
        return res.view(dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )[:, slicing.frame_offset:slicing.frame_offset + slicing.frame_size]


class DMFileSet(FileSet):
    pass


class DMDataSet(DataSet):
    """
    Factory class for DigitalMicrograph file datasets

     - Passing either the :code:`files` kwarg or a tuple/list as first
       argument will create an instance of :class:`StackedDMDataSet`
     - Passing either the :code:`path` kwarg or any other object
       as first argument will create a :class:`SingleDMDataSet`

    This class is necessary to handle the difference in signatures and
    behaviours of the two DM dataset implementations, but these may
    later be fused if a markup format for multi-file datasets is implemented

    This class implements the methods necessary to expose a DMDataSet in
    the web GUI, which it does by deferring to SingleDMDataSet. At this
    time multi-file datasets are not supported in the UI.

    NOTE this way of generating the subclasses breaks deeper
    subclassing, as this __new__ method will always instantiate
    a SingleDMDataSet or StackedDMDataSet, and not a subclass of
    either of these. This could potentially be improved by using
    the .__instance_subclass__() staticmethod to register the
    subclasses and what they inherit from.
    """
    def __new__(cls, *args, **kwargs):
        # delayed here to avoid circular reference
        from .dm_single import SingleDMDataSet
        if 'path' in kwargs:
            subclass = SingleDMDataSet
        elif 'files' in kwargs:
            subclass = StackedDMDataSet
        elif args and isinstance(args[0], (list, tuple)):
            subclass = StackedDMDataSet
        else:
            subclass = SingleDMDataSet
        return super().__new__(subclass)

    @classmethod
    def get_supported_extensions(cls):
        return {"dm3", "dm4"}

    @classmethod
    def get_msg_converter(cls):
        return SingleDMDatasetParams

    @classmethod
    def detect_params(cls, path, executor):
        # delayed here to avoid circular reference
        from .dm_single import SingleDMDataSet
        return SingleDMDataSet.detect_params(path, executor)


class StackedDMDataSet(DMDataSet):
    """
    Reader for stacks of DM3/DM4 files.

    Note
    ----
    This DataSet is not supported in the GUI yet, as the file dialog needs to be
    updated to `properly handle opening series
    <https://github.com/LiberTEM/LiberTEM/issues/498>`_.

    Note
    ----
    Single-file 3/4D DM datasets are supported through the
    :class:`~libertem.io.datasets.dm_single.SingleDMDataSet` class.

    Note
    ----
    You can use the PyPI package `natsort <https://pypi.org/project/natsort/>`_
    to sort the filenames by their numerical components, this is especially useful
    for filenames without leading zeros.

    Parameters
    ----------

    files : List[str]
        List of paths to the files that should be loaded. The order is important,
        as it determines the order in the navigation axis.

    nav_shape : Tuple[int, ...] or None
        By default, the files are loaded as a 3D stack. You can change this
        by specifying the nav_shape, which reshapes the navigation dimensions.
        Raises a `DataSetException` if the shape is incompatible with the data
        that is loaded.

    sig_shape: Tuple[int, ...], optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    same_offset : bool
        When reading a stack of dm3/dm4 files, it can be expensive to read in
        all the metadata from all files, which we currently only use for
        getting the offsets and sizes of the main data in each file. If you
        absolutely know that the offsets and sizes are the same for all files,
        you can set this parameter and we will skip reading all metadata but
        the one from the first file.

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.
    """
    def __init__(self, files=None, scan_size=None, same_offset=False, nav_shape=None,
                 sig_shape=None, sync_offset=0, io_backend=None, num_partitions=None):
        super().__init__(io_backend=io_backend, num_partitions=num_partitions)
        self._meta = None
        self._same_offset = same_offset
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        # handle backwards-compatability:
        if scan_size is not None:
            warnings.warn(
                "scan_size argument is deprecated. please specify nav_shape instead",
                FutureWarning
            )
            if nav_shape is not None:
                raise ValueError("cannot specify both scan_size and nav_shape")
            self._nav_shape = tuple(scan_size)
        self._filesize = None
        self._files = files
        if not isinstance(files, (list, tuple)):
            raise DataSetException("files argument must be an iterable\
                                    of file paths, recieved {type(files)}")
        if len(files) == 0:
            raise DataSetException("need at least one file as input!")
        self._fileset = None
        # per-file cached attributes:
        self._z_sizes = {}
        self._offsets = {}

    def __new__(cls, *args, **kwargs):
        '''
        Skip the superclasse's :code:`__new__()` method.

        Instead, go straight to the grandparent. That disables the
        :class:`DMDataSet` type determination magic. Otherwise unpickling will
        always yield a :class:`SingleDMDataSet` since this class inherits the
        parent's :code:`__new__()` method and unpickling calls it without
        parameters, making it select :class:`SingleDMDataSet`.

        It mimics calling the superclass :code:`__new__(cls)` without additional
        parameters, just like the parent's method.
        '''
        return DataSet.__new__(cls)

    def _get_sig_shape_and_native_dtype(self):
        first_fn = self._get_files()[0]
        first_file = fileDM(first_fn, on_memory=True)
        if first_file.numObjects == 1:
            idx = 0
        else:
            idx = 1
        try:
            raw_dtype = first_file._DM2NPDataType(first_file.dataType[idx])
            native_sig_shape = (first_file.ySize[idx], first_file.xSize[idx])
        except IndexError as e:
            raise DataSetException("could not determine dtype or signal shape") from e
        return native_sig_shape, raw_dtype

    def _get_fileset(self):
        start_idx = 0
        files = []
        for fn in self._get_files():
            z_size = self._z_sizes[fn]
            f = StackedDMFile(
                path=fn,
                start_idx=start_idx,
                end_idx=start_idx + z_size,
                sig_shape=self._meta.shape.sig,
                native_dtype=self._meta.raw_dtype,
                file_header=self._offsets[fn],
            )
            files.append(f)
            start_idx += z_size
        return DMFileSet(files)

    def _get_files(self):
        return self._files

    def _get_filesize(self):
        return sum(
            os.stat(p).st_size
            for p in self._get_files()
        )

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_filesize)
        if self._same_offset:
            metadata = executor.run_function(_get_metadata, self._get_files()[0])
            self._offsets = {
                fn: metadata['offset']
                for fn in self._get_files()
            }
            self._z_sizes = {
                fn: metadata['zsize']
                for fn in self._get_files()
            }
        else:
            metadata = dict(zip(
                self._get_files(),
                executor.map(_get_metadata, self._get_files()),
            ))
            self._offsets = {
                fn: metadata[fn]['offset']
                for fn in self._get_files()
            }
            self._z_sizes = {
                fn: metadata[fn]['zsize']
                for fn in self._get_files()
            }
        self._image_count = int(sum(self._z_sizes.values()))
        if self._nav_shape is None:
            self._nav_shape = (sum(self._z_sizes.values()),)
        native_sig_shape, native_dtype = executor.run_function(self._get_sig_shape_and_native_dtype)
        if self._sig_shape is None:
            self._sig_shape = tuple(native_sig_shape)
        elif int(prod(self._sig_shape)) != int(prod(native_sig_shape)):
            raise DataSetException(
                "sig_shape must be of size: %s" % int(prod(native_sig_shape))
            )
        shape = self._nav_shape + self._sig_shape
        self._nav_shape_product = int(prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=Shape(shape, sig_dims=len(self._sig_shape)),
            raw_dtype=native_dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        self._fileset = executor.run_function(self._get_fileset)
        return self

    @classmethod
    def get_supported_extensions(cls):
        return {"dm3", "dm4"}

    @classmethod
    def get_msg_converter(cls) -> type[MessageConverter]:
        return StackedDMDatasetParams

    @classmethod
    def detect_params(cls, path, executor):
        # FIXME: this doesn't really make sense for file series
        # pl = path.lower()
        # if pl.endswith(".dm3") or pl.endswith(".dm4"):
        #     return {
        #         "parameters": {
        #             "files": [path]
        #         },
        #     }
        return False

    @property
    def dtype(self) -> "nt.DTypeLike":
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        first_fn = self._get_files()[0]
        try:
            with fileDM(first_fn, on_memory=True):
                pass
            return True
        except OSError as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        for part_slice, start, stop in self.get_slices():
            yield BasePartition(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=self._fileset,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<DMDataSet for a stack of %d files>" % (len(self._get_files()),)
