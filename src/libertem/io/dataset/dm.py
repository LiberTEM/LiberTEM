import os
import typing
from typing import Dict, Any, Tuple, Optional
import logging
import warnings

from ncempy.io.dm import fileDM
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape, Slice
from libertem.io.dataset.base.file import OffsetsSizes
from libertem.common.messageconverter import MessageConverter
from libertem.io.dataset.base.tiling import DataTile
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta, File,
)
from libertem.io.dataset.base.backend_fortran import FortranReader

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from numpy import typing as nt
    from libertem.io.dataset.base.tiling_scheme import TilingScheme


class DMDatasetParams(MessageConverter):
    SCHEMA: typing.Dict = {}

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
    Reader for stacks of DM3/DM4 files.

    Note
    ----
    This DataSet is not supported in the GUI yet, as the file dialog needs to be
    updated to `properly handle opening series
    <https://github.com/LiberTEM/LiberTEM/issues/498>`_.

    Note
    ----
    Single-file 4D DM files are not yet supported. The use-case would be
    to read DM4 files from the conversion of K2 STEMx data, but those data sets
    are actually transposed (nav/sig are swapped).

    That means the data would have to be transposed back into the usual shape,
    which is slow, or algorithms would have to be adapted to work directly on
    transposed data. As an example, applying a mask in the conventional layout
    corresponds to calculating a weighted sum frame along the navigation
    dimension in the transposed layout.

    Since the transposed layout corresponds to a TEM tilt series, support for
    transposed 4D STEM data could have more general applications beyond
    supporting 4D DM4 files. Please contact us if you have a use-case for
    single-file 4D DM files or other applications that process stacks of TEM
    files, and we may add support!

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

    nav_shape : Tuple[int] or None
        By default, the files are loaded as a 3D stack. You can change this
        by specifying the nav_shape, which reshapes the navigation dimensions.
        Raises a `DataSetException` if the shape is incompatible with the data
        that is loaded.

    sig_shape: Tuple[int], optional
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
    """
    def __init__(self, files=None, scan_size=None, same_offset=False, nav_shape=None,
                 sig_shape=None, sync_offset=0, io_backend=None):
        super().__init__(io_backend=io_backend)
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
        self._image_count = sum(self._z_sizes.values())
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
    def get_msg_converter(cls) -> typing.Type[MessageConverter]:
        return DMDatasetParams

    @classmethod
    def detect_params(cls, path, executor):
        # FIXME: this doesn't really make sense for file series
        pl = path.lower()
        if pl.endswith(".dm3") or pl.endswith(".dm4"):
            return {
                "parameters": {
                    "files": [path]
                },
            }
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


class SingleDMFileDataset(DataSet):
    """
    Reader for a single DM3/DM4 file

    Where possible the navigation structure: 4D-scan or 3D-stack,
    will be inferred from the file metadata, but this can be overridden
    using the arguments. In the GUI a 3D-stack will have an extra
    singleton navigation dimension prepended to allow it to display.

    Note
    ----
    Single-file 4D DM(4) files are only supported where they were recorded
    using C-ordering, which is an option in recent versions of GMS.
    This dataset will try to infer if this is the true from the file
    tags and throw a warning, but depending on the tag integrity
    this might not be possible.

    Any UDF results for an F-ordered DM file will be invalid.

    Parameters
    ----------

    path : PathLike
        The path to the .dm3/.dm4 file

    nav_shape : Tuple[int] or None
        By default, the files are loaded as a 3D stack. You can change this
        by specifying the nav_shape, which reshapes the navigation dimensions.
        Raises a `DataSetException` if the shape is incompatible with the data
        that is loaded.

    sig_shape: Tuple[int], optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    """
    def __init__(self, path, nav_shape=None, sig_shape=None,
                 sync_offset=0, io_backend=None, sig_dims=2):
        super().__init__(io_backend=io_backend)
        self._filesize = None

        self._path = path
        self._nav_shape = tuple(nav_shape) if nav_shape else None
        self._sig_shape = tuple(sig_shape) if sig_shape else None
        self._sync_offset = sync_offset
        self._sig_dims = sig_dims
        if self._sig_dims != 2:
            raise DataSetException('Non-2D signals not yet supported in SingleDMFileDataset')

    def __repr__(self):
        return f"<SingleDMFileDataset {self._path}>"

    @property
    def dtype(self) -> "nt.DTypeLike":
        return self.meta.raw_dtype

    @property
    def shape(self):
        return self.meta.shape

    @classmethod
    def get_supported_extensions(cls):
        return {"dm3", "dm4"}

    def _get_filesize(self):
        return os.stat(self._path).st_size

    @classmethod
    def get_msg_converter(cls) -> typing.Type[MessageConverter]:
        ...

    @classmethod
    def detect_params(cls, path, executor):
        ...

    def check_valid(self):
        try:
            with fileDM(self._path, on_memory=True):
                pass
            return True
        except OSError as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _read_metadata(self):
        with fileDM(self._path, on_memory=True) as fp:
            array_map = {}
            ndims = fp.dataShape
            for ds_idx, dims in enumerate(ndims):
                if dims < 3:
                    # Single-image DM3/4 files not supported
                    continue
                shape = (fp.xSize[ds_idx], fp.ySize[ds_idx], fp.zSize[ds_idx])
                if dims == 4:
                    shape = shape + (fp.zSize2[ds_idx],)
                array_map[ds_idx] = {'shape': shape, 'ds_idx': ds_idx}
            if not array_map:
                raise DataSetException('Unable to find any 3/4D datasets in DM file')
            elif len(array_map) > 1:
                raise DataSetException('No support for multiple datasets in same file')

            array_offsets = fp.dataOffset
            array_sizes = fp.dataSize
            array_types = fp.dataType
            for ds_idx in array_map.keys():
                array_map[ds_idx]['offset'] = array_offsets[ds_idx]
                array_map[ds_idx]['size'] = array_sizes[ds_idx]
                try:
                    array_map[ds_idx]['dtype'] = fp._DM2NPDataType(array_types[ds_idx])
                except OSError:
                    # unconvertible DM data type
                    array_map[ds_idx]['dtype'] = array_types[ds_idx]

            tags = fp.allTags
        assert len(array_map) == 1
        array_meta = array_map[ds_idx]
        return array_meta, tags

    @staticmethod
    def _tags_to_nest(tags: Dict[str, Any]):
        tags_nest = {}
        for tag, element in tags.items():
            tag = tag.strip('.')
            _insert_to = tags_nest
            for tag_el in tag.split('.')[:-1]:
                try:
                    _insert_to = _insert_to[tag_el]
                except KeyError:
                    _insert_to[tag_el] = {}
                    _insert_to = _insert_to[tag_el]
            _insert_to[tag.split('.')[-1]] = element
        return tags_nest

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_filesize)
        array_meta, all_tags = executor.run_function(self._read_metadata)
        # nest = self._tags_to_nest(all_tags)
        # dm_data_key = str(array_meta['ds_idx'] + 1)
        # data_tags = nest['ImageList'][dm_data_key]
        self._array_offset = array_meta['offset']
        self._raw_dtype = array_meta['dtype']
        assert self._raw_dtype is not None and not isinstance(self._raw_dtype, int)

        # The shape reversal to read in C-ordering applies to DM4/STEM files
        # saved in the new style as well as DM3, 3D image stacks saved
        # in older versions of GMS. Must check whether newer image stacks
        # are saved in C-ordering as well (despite the metadata order)
        self._array_c_ordered = False
        if self._array_c_ordered:
            _shape = tuple(reversed(array_meta['shape']))
        else:
            # Data on disk is stored as (flat_sig_c_order, flat_nav_c_order)
            # so a hybrid C/F byte order, as such we need to do some
            # gymnastics to get a normal LT shape
            _shape = (array_meta['shape'][:-self._sig_dims][::-1]
                      + array_meta['shape'][-self._sig_dims:][::-1])
            # NOTE it should be possible to apply sync_offset
            # as the nav dimension is fundamentally c-ordered on disk
            # but at this time the lack of use cases and the complexity
            # it would bring means enforcing self._sync_offset == 0
            # in these style of DM4 files is preferred for now
            assert self._sync_offset == 0
        self._nav_shape = _shape[:-self._sig_dims]
        self._sig_shape = _shape[-self._sig_dims:]
        # regardless of file order the Dataset shape property is 'standard'

        # this might not be how sync_offset works!
        # _array_shape = (1, 1, 1, 1)
        # _array_size = int(prod(_array_shape))
        # _specified_size= int(prod(self._nav_shape + self._sig_shape))
        # if _specified_size > _array_size:
        #     raise DataSetException(
        #         "specified shape must be <= data in file: "
        #         f"array in file {_array_shape}, {_array_size} elements, "
        #         f"specified shape {self._nav_shape + self._sig_shape}, {_specified_size} elements"
        #     )

        self._nav_shape_product = int(prod(self._nav_shape))
        # Can't infer image_count from filesize, must assume it matches the nav_shape
        # Need to go understand sync_offset/image_count/nav_shape_product yet again
        self._image_count = self._nav_shape_product
        self._sync_offset_info = self.get_sync_offset_info()
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._raw_dtype),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def _get_fileset(self):
        return DMFileSet([
            DMFile(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
                sig_shape=self.shape.sig,
                native_dtype=self.meta.raw_dtype,
                file_header=self._array_offset,
            )
        ])

    # def get_num_partitions(self):
    #     """
    #     When C-ordered defer to the default behaviour

    #     F-ordering encourages very deep tile stacks,
    #     ideally depth == ds.meta.nav_shape_product,
    #     which can only be acheived with a single partition.
    #     As for now we can't do overlapping partitions sliced
    #     in the sig dimensions, the best case is to have very few
    #     partitions each generating very short, single-sig-row tiles
    #     The reality is that each partition will have to memmap and
    #     traverse the whole file many times, so minimising the length
    #     of any partiticular traversal is 'optimal'.

    #     As all the data are disjoint anyway, to maintain the API
    #     we absolutely must perform a copy to re-order to C before yielding tiles

    #     This poses a problem for any `process_frame` or `process_partition`
    #     UDF, which could end up loading a very large amount of data into memory.

    #     For certain tasks (particular ApplyMasksUDF and SumUDF) it is
    #     in principle optimal to work on F-ordered data, but the API
    #     changes to make this transparent for user-facing code are monumental
    #     """

    def get_partitions(self):
        partition_cls = DMPartition if self._array_c_ordered else DMPartitionFortran
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield partition_cls(
                meta=self.meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def adjust_tileshape(
        self, tileshape: Tuple[int, ...], roi: Optional[np.ndarray]
    ) -> Tuple[int, ...]:
        """
        If C-ordered, return proposed tileshape
        If hybrid C/F ordered adjust tileshape so that
        only the first signal dimension (rows) is tiled
        as this should incurr reads in blocks of complete rows
        without incurring additional striding

        # NOTE Check how corrections could be broken ??
        """
        if self._array_c_ordered:
            return tileshape
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


class DMFile(File):
    ...


class DMPartition(BasePartition):
    ...


class RawPartitionFortran(BasePartition):
    sig_order = 'F'

    def get_tiles(self, tiling_scheme: 'TilingScheme', dest_dtype="float32", roi=None):
        if self._start_frame >= self.meta.image_count:
            return

        assert len(self._fileset) == 1
        file: DMFile = self._fileset[0]

        dest_dtype = np.dtype(dest_dtype)
        tiling_scheme_adj = tiling_scheme.adjust_for_partition(self)
        self.validate_tiling_scheme(tiling_scheme_adj)

        reader = FortranReader(
                        file.path,
                        self.meta.shape,
                        self.meta.raw_dtype,
                        tiling_scheme_adj,
                        sig_order=self.sig_order,
                        file_header=file.file_header_bytes,
                    )
        reader.create_memmaps()

        part_slice = slice(self._start_frame, self._start_frame + self._num_frames)
        has_roi = roi is not None
        if has_roi:
            read_range = np.flatnonzero(roi.reshape(-1, order=self.shape.nav_order))
            read_indices = np.arange(read_range.size, dtype=np.int64)
            part_mask = np.logical_and(read_range >= part_slice.start,
                                       read_range < part_slice.stop)
            read_range = read_range[part_mask]
            assert read_range.size, 'Empty ROI part ??'
            read_range_idcs = read_indices[part_mask]
            index_lookup = {full: compressed for full, compressed
                            in zip(read_range, read_range_idcs)}
        else:
            read_range = (part_slice,)

        for frame_idcs, scheme_idx, tile in reader.generate_tiles(*read_range):
            # frame_idcs is tuple(int, ...) of f-ordered, whole-dataset indices
            # not including compression for ROI (same as for read_range arg)
            scheme_slice = tiling_scheme_adj[scheme_idx]
            if has_roi:
                nav_origin = (index_lookup[frame_idcs[0]],)
            else:
                nav_origin = (frame_idcs[0],)
            tile_slice = Slice(
                origin=nav_origin + scheme_slice.origin,
                shape=tile.shape[:1] + scheme_slice.shape,
            )
            if dest_dtype != tile.dtype:
                tile = tile.astype(dest_dtype)
            self.preprocess(tile, tile_slice, self._corrections)
            yield DataTile(
                tile,
                tile_slice=tile_slice,
                scheme_idx=scheme_idx,
            )

    def preprocess(self, data, tile_slice, corrections):
        if corrections is None:
            return
        corrections.apply(data, tile_slice)


class DMPartitionFortran(RawPartitionFortran):
    sig_order = 'C'
