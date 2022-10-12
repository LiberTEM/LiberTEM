import os
import typing
from typing import Dict, Any, Tuple, Optional
import logging

import ncempy.io.dm
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape, Slice
from libertem.common.messageconverter import MessageConverter
from libertem.io.dataset.base.tiling import DataTile
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta,
    File, IOBackend,
)
from libertem.io.dataset.base.backend_fortran import FortranReader

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from numpy import typing as nt
    from libertem.io.dataset.base.tiling_scheme import TilingScheme
    from libertem.executor.base import JobExecutor
    from libertem.io.corrections.corrset import CorrectionSet


class DM4DatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/DM4DatasetParams.schema.json",
        "title": "DM4DatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "DM4"},
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
        },
        "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        for k in ["nav_shape", "sig_shape", "sync_offset"]:
            if k in raw_data:
                data[k] = raw_data[k]
        return data


class DM4DataSet(DataSet):
    """
    Reader for a single DM3/DM4 file

    Where possible the navigation structure: 4D-scan or 3D-stack,
    will be inferred from the file metadata. In the GUI a 3D-stack
    will have an extra singleton navigation dimension prepended
    to allow it to display. DM files containing only a single data array
    are supported (multi-array files will raise an error).

    Note
    ----
    Single-file 4D DM(4) files are can be stored using normal C-ordering,
    which is an option in recent versions of GMS. In this case the dataset
    will try to infer this and use normal LiberTEM I/O for reading the file.
    If the file uses the older hybrid C/F-ordering (flat_sig, flat_nav)
    then a dedicated I/O backend will be used instead, though performance
    will be severely limited in this mode.

    The format is normally stored in the DM tags, but depending on the tag
    integrity it might not be possible to infer it. The DM-default is hybrid
    C/F-ordering, therefore the lower-performance specialised I/O is
    used by default. A C-ordered format interpretation can be forced
    using the dataset parameters.

    Parameters
    ----------

    path : PathLike
        The path to the .dm3/.dm4 file

    nav_shape : Tuple[int] or None
        FIXME

    sig_shape: Tuple[int], optional
        FIXME

    sync_offset: int, optional, by default 0
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
        Only allowable != 0 if the file is C-ordered

    force_c_order: bool, optional, by default False
        Force the data to be interpreted as a C-ordered
        array regardless of the tag information. This will lead
        to incorrect results on an hybrid C/F-ordered file
    """
    def __init__(self, path, nav_shape=None, sig_shape=None,
                 sync_offset=0, io_backend=None, sig_dims=2,
                 force_c_order=False):
        super().__init__(io_backend=io_backend)
        self._filesize = None

        self._path = path
        self._nav_shape = tuple(nav_shape) if nav_shape else None
        self._sig_shape = tuple(sig_shape) if sig_shape else None
        self._sync_offset = sync_offset
        self._sig_dims = sig_dims
        self._force_c_order = force_c_order
        if self._sig_dims != 2:
            raise DataSetException('Non-2D signals not yet supported in DM4DataSet')

    def __repr__(self):
        try:
            shape = f' - {self.shape}'
        except AttributeError:
            shape = ''
        return f"<DMFileDataset {self._path}>" + shape

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
    def get_msg_converter(cls):
        return DM4DatasetParams

    @classmethod
    def detect_params(cls, path, executor):
        pathlow = path.lower()
        if pathlow.endswith(".dm3") or pathlow.endswith(".dm4"):
            array_meta = executor.run_function(cls._read_metadata, path)
            sync_offset = 0
            nav_shape, sig_shape = cls._modify_shape(array_meta['shape'],
                                                     array_meta['c_order'])
            if len(nav_shape) == 1:
                nav_shape = (1,) + nav_shape
            image_count = prod(nav_shape)
        else:
            return False
        return {
            "parameters": {
                "path": path,
                "nav_shape": nav_shape,
                "sig_shape": sig_shape,
                "sync_offset": sync_offset,
            },
            "info": {
                "image_count": image_count,
                "native_sig_shape": sig_shape,
            }
        }

    def check_valid(self):
        try:
            with ncempy.io.dm.fileDM(self._path, on_memory=True):
                pass
            return True
        except OSError as e:
            raise DataSetException("invalid dataset: %s" % e)

    @classmethod
    def _read_metadata(cls, path):
        with ncempy.io.dm.fileDM(path, on_memory=True) as fp:
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
            array_types = fp.dataType
            for ds_idx in array_map.keys():
                array_map[ds_idx]['offset'] = array_offsets[ds_idx]
                try:
                    array_map[ds_idx]['dtype'] = fp._DM2NPDataType(array_types[ds_idx])
                except OSError:
                    # unconvertible DM data type
                    array_map[ds_idx]['dtype'] = array_types[ds_idx]

            tags = fp.allTags
        assert len(array_map) == 1
        array_meta = array_map[ds_idx]

        # Infer array ordering
        nest = cls._tags_to_nest(tags)
        dm_data_key = str(array_meta['ds_idx'] + 1)
        data_tags = nest['ImageList'][dm_data_key]
        try:
            # Must bool(int(val)) just in case the flag is string '0'
            is_c_ordered = bool(int(data_tags['ImageTags']['Meta Data']['Data Order Swapped']))
        except (KeyError, ValueError):
            # Defer to tag, otherwise assume F-order unless 3D dataset which seems to be C-ordered
            is_c_ordered = False or (len(array_meta['shape']) == 3)
        array_meta['c_order'] = is_c_ordered
        return array_meta

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

    @staticmethod
    def _modify_shape(shape: Tuple[int], c_order: bool, sig_dims: int = 2):
        # The shape reversal to read in C-ordering applies to DM4/STEM files
        # saved in the new style as well as DM3, 3D image stacks saved
        # in older versions of GMS. Must check whether newer image stacks
        # are saved in C-ordering as well (despite the metadata order)
        if c_order:
            shape = tuple(reversed(shape))
        else:
            # Data on disk is stored as (flat_sig_c_order, flat_nav_c_order)
            # so a hybrid C/F byte order, as such we need to do some
            # gymnastics to get a normal LT shape
            shape = (shape[:-sig_dims][::-1]
                     + shape[-sig_dims:][::-1])
        shape = tuple(map(int, shape))
        nav_shape = shape[:-sig_dims]
        sig_shape = shape[-sig_dims:]
        return nav_shape, sig_shape

    def initialize(self, executor: 'JobExecutor'):
        self._filesize = executor.run_function(self._get_filesize)
        array_meta = executor.run_function(self._read_metadata, self._path)
        self._array_offset = array_meta['offset']
        self._raw_dtype = array_meta['dtype']
        assert self._raw_dtype is not None and not isinstance(self._raw_dtype, int)
        self._array_c_ordered = self._force_c_order or array_meta['c_order']
        nav_shape, sig_shape = self._modify_shape(array_meta['shape'],
                                                  self._array_c_ordered,
                                                  sig_dims=self._sig_dims)

        # Image count is true number of frames in file (?)
        self._image_count = int(prod(nav_shape))
        if self._nav_shape is not None:
            manual_nav_shape_product = prod(self._nav_shape)
            if manual_nav_shape_product > self._image_count:
                raise DataSetException('Specified nav_shape greater than file nav size')
            elif self._array_c_ordered and (manual_nav_shape_product != self._image_count):
                raise DataSetException('Manual nav shape with different size '
                                       'not supported for F-ordered DM4')
        else:
            self._nav_shape = nav_shape
        # nav_shape product is either manual nav_shape if supplied or metadata nav_shape (?)
        self._nav_shape_product = int(prod(self._nav_shape))
        sig_size = int(prod(sig_shape))
        if self._sig_shape is not None:
            manual_sig_size = int(prod(self._sig_shape))
            if self._array_c_ordered and (sig_size != manual_sig_size):
                raise DataSetException('Manual sig shape with different size '
                                       'not supported for F-ordered DM4')
            elif (manual_sig_size * self._nav_shape_product) > (self._image_count * sig_size):
                raise DataSetException('Specified sig_shape and nav size '
                                       'too large for data in file')
        else:
            self._sig_shape = sig_shape

        # regardless of file order the Dataset shape property is 'standard'
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._raw_dtype),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def _get_fileset(self):
        return DM4FileSet([
            DM4File(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
                sig_shape=self.shape.sig,
                native_dtype=self.meta.raw_dtype,
                file_header=self._array_offset,
            )
        ])

    def get_partitions(self):
        partition_cls = DM4Partition if self._array_c_ordered else DM4PartitionFortran
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

    def need_decode(
        self,
        read_dtype: "nt.DTypeLike",
        roi: Optional[np.ndarray],
        corrections: Optional['CorrectionSet'],
    ) -> bool:
        if self._array_c_ordered:
            return super().need_decode(read_dtype, roi, corrections)
        else:
            # Not strictly true but the strange layout means
            # read performance is the limiting factor
            return True


class DM4FileSet(FileSet):
    pass


class DM4File(File):
    ...


class DM4Partition(BasePartition):
    ...


class RawPartitionFortran(BasePartition):
    sig_order = 'F'

    def get_tiles(self, tiling_scheme: 'TilingScheme', dest_dtype="float32", roi=None):
        if self._start_frame >= self.meta.image_count:
            return

        assert len(self._fileset) == 1
        file: DM4File = self._fileset[0]

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

        sync_offset = self.meta.sync_offset
        ds_size = self.meta.shape.nav.size

        # Define the frame slice to read in this partition for frames which actually exist
        part_slice = slice(max(0, self._start_frame),
                           min(ds_size, self._start_frame + self._num_frames))
        has_roi = roi is not None
        if has_roi:
            # Must unroll the ROI in the same way as the partitions were
            # created, i.e. C- or F- unrolling, held by the shape.nav_order property
            # read_range is real frame indices in the array
            # the roi is shifted by sync_offset to get the true frames to read
            read_range = np.flatnonzero(roi.reshape(-1)) + sync_offset
            # read_indices are in the normal flat navigation space
            # implicitly sync_offset applies to these indices therefore
            # no need to shift the slices later to match
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
            # frame_idcs is tuple(int, ...) of whole-dataset indices
            # not including compression for ROI (same as for read_range arg)
            # the meaning of the whole-dataset indices depends on if the
            # nav dimension was unrolled in F or C-ordering
            # these frame_idcs are are the frames actually read from the
            # array in the file as if sync_offset did not exist
            scheme_slice = tiling_scheme_adj[scheme_idx]
            if has_roi:
                nav_origin = (index_lookup[frame_idcs[0]],)
            else:
                # apply slice_offset to shift origin for sync_offset
                nav_origin = (frame_idcs[0] - sync_offset,)
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


class DM4PartitionFortran(RawPartitionFortran):
    sig_order = 'C'
