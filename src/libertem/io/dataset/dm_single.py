import os
import warnings
import typing
from typing import Dict, Any, Tuple, Optional
import logging

import ncempy.io.dm
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape, Slice
from libertem.io.dataset.base.tiling import DataTile
from .base import BasePartition, DataSetException, DataSetMeta, File
from .dm import DMDataSet, SingleDMDatasetParams, DMFileSet
from .dm_single_reader import FortranReader

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from numpy import typing as nt
    from libertem.io.dataset.base.tiling_scheme import TilingScheme
    from libertem.executor.base import JobExecutor
    from libertem.io.corrections.corrset import CorrectionSet


class SingleDMDataSet(DMDataSet):
    """
    Reader for a single DM3/DM4 file. Handles 4D-STEM, 3D-Spectrum Images,
    and TEM image stacks stored in a single-file format. Where possible
    the structure will be inferred from the file metadata.

    Note
    ----
    Single-file DM data can be stored on disk using either normal C-ordering,
    which is an option in recent versions of GMS, or an alternative F/C-hybrid
    ordering depending on the imaging mode and dimensionality.

    The DataSet will try to infer the ordering from the file metadata and
    read accordingly. If the file uses the older hybrid F/C-ordering
    :code:`(flat_sig, flat_nav)`, then a dedicated I/O backend will be used,
    and performance may be severely limited compared to 'standard' files.

    A C-ordered interpretation can be forced using the `force_c_order` argument.

    Note
    ----
    In the Web-GUI a 2D-image or 3D-stack/spectrum image will have extra
    singleton navigation dimensions prepended to allow them to display.
    DM files containing multiple datasets are supported via the
    `dataset_index` argument.

    While capable of reading 2D/3D files, LiberTEM is not particularly
    well-adapted to processing these data and the user should consider
    other tools. Individual spectra or vectors (1D data) are not supported.

    Parameters
    ----------

    path : PathLike
        The path to the .dm3/.dm4 file

    nav_shape : Tuple[int], optional
        Over-ride the nav_shape provided by the file metadata.
        In new-style DM files, this can be used to adjust the total
        number of frames, while in old-style files only reshaping
        while maintaining the same number of frames is possible.

    sig_shape: Tuple[int], optional
        Over-ride the sig_shape provided by the file metadata.
        Data are read sequentially in all cases, therefore this
        is typically only interesting if the total number of
        sig pixels remains constant.

    sync_offset: int, optional, by default 0
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    io_backend: IOBackend, optional
        A specific IOBackend implementation to over-ride the
        platform default. Note, for F/C-ordered (old-style)
        files cannot be read with standard IOBackends, and therefore
        this argument will be ignored.

    force_c_order: bool, optional, by default False
        Force the data to be interpreted as a C-ordered
        array regardless of the tag information. This will lead
        to incorrect results on an hybrid C/F-ordered file

    dataset_index: int, optional
        In the case of a multi-dataset DM file this can
        be used to open a specific dataset index. Note that
        the datasets in a DM-file often begin with a thumbnail
        which occupies the 0 dataset index. If not provided the
        first compatible dataset found in the file is used.
    """
    def __init__(self, path, nav_shape=None, sig_shape=None,
                 sync_offset=0, io_backend=None,
                 force_c_order=False, dataset_index=None):
        super().__init__(io_backend=io_backend)
        self._filesize = None

        self._path = path
        self._nav_shape = tuple(nav_shape) if nav_shape else None
        self._sig_shape = tuple(sig_shape) if sig_shape else None
        self._sync_offset = sync_offset
        self._force_c_order = force_c_order
        self._dm_ds_index = dataset_index

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
        return SingleDMDatasetParams

    @classmethod
    def detect_params(cls, path: str, executor):
        pathlow = path.lower()
        if pathlow.endswith(".dm3") or pathlow.endswith(".dm4"):
            array_meta = executor.run_function(cls._read_metadata, path)
            sig_dims = array_meta['sig_dims']
            if sig_dims == 1:
                # raise ValueError('Unable to display 1D spectra as signals in the GUI')
                return False
            sync_offset = 0
            nav_shape, sig_shape = cls._modify_shape(array_meta['shape'],
                                                    array_meta['c_order'],
                                                    sig_dims=sig_dims)
            if len(nav_shape) == 1:
                nav_shape = (1,) + nav_shape
            image_count = prod(nav_shape)
        else:
            # raise ValueError(f'Unecognized extension {os.path.splitext(pathlow)[1]}')
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
    def _read_metadata(cls, path, use_ds=None):
        with ncempy.io.dm.fileDM(path, on_memory=True) as fp:
            tags = fp.allTags
            array_map = {}
            start_from = 1 if fp.thumbnail else 0
            for ds_idx in range(start_from, fp.numObjects):
                dims = fp.dataShape[ds_idx]
                if dims < 2:
                    # Spectrum-only ?
                    continue
                shape = (fp.xSize[ds_idx], fp.ySize[ds_idx])
                if dims > 2:
                    shape = shape + (fp.zSize[ds_idx],)
                if dims > 3:
                    shape = shape + (fp.zSize2[ds_idx],)
                array_map[ds_idx] = {'shape': shape, 'ds_idx': ds_idx}
                array_map[ds_idx]['offset'] = fp.dataOffset[ds_idx]
                try:
                    array_map[ds_idx]['dtype'] = fp._DM2NPDataType(fp.dataType[ds_idx])
                except OSError:
                    # unconvertible DM data type
                    array_map[ds_idx]['dtype'] = fp.dataType[ds_idx]

        if not array_map:
            raise DataSetException('Unable to find any 2/3/4D datasets in DM file')

        if use_ds is not None:
            if use_ds in array_map.keys():
                ds_idx = use_ds
            else:
                raise DataSetException(f'Specified dataset idx {use_ds} not found in file')
        else:
            # Use first dataset index we loaded
            ds_idx = [*array_map.keys()][0]
            if len(array_map) > 1:
                warnings.warn(
                    "Found multiple datasets in DM file, using first dataset",
                    RuntimeWarning
                )

        array_meta = array_map[ds_idx]
        ndims = len(array_meta['shape'])
        # Set default metadata in case tags are incomplete
        array_meta['format'] = 'Unknown'
        array_meta['sig_dims'] = 2
        # Assume C-ordering for 2D images and 3D image stacks
        # Assume F-ordering for STEM data unless tagged otherwise
        # Spectrum images are also F-ordered but these data must
        # be recognized from the tags (they can be 2- or 3-D)
        array_meta['c_order'] = True if ndims in (2, 3) else False

        # Infer array ordering
        nest = cls._tags_to_nest(tags)
        # Must + 1 because DM uses 1-based-indexing in its tags
        dm_data_key = str(array_meta['ds_idx'] + 1)
        try:
            data_tags = nest['ImageList'][dm_data_key]['ImageTags']
        except KeyError:
            # unrecognized / invalid tag structure, return defaults
            return array_meta

        if 'Meta Data' in data_tags:
            meta_data = data_tags['Meta Data']
            array_meta['format'] = meta_data.get('Format', 'Unknown')
            if str(array_meta['format']).strip().lower() == 'spectrum image':
                assert ndims in (2, 3)
                array_meta['sig_dims'] = 1
                if ndims == 3:
                    # 3-D spectrum images seem to be F-ordered
                    # 2-D SI are seemingly C-ordered (value set above)
                    array_meta['c_order'] = False
            if 'Data Order Swapped' in meta_data:
                # Always defer to tag for ordering if available
                # This line handes the new-style STEM datasets
                # The bool(int()) is just-in-case for string tags
                array_meta['c_order'] = bool(int(meta_data['Data Order Swapped']))

        # Need to find a 3D image stack with the 'Meta Data' + 'Format' tags
        if array_meta['format'] not in ('Spectrum image',
                                        'Image',
                                        'Diffraction image'):
            warnings.warn(
                f"Unrecognized image format {array_meta['format']}, "
                "DM tags may be parsed incorrectly",
                RuntimeWarning
            )
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
        if not nav_shape:
            # Special case for 2D image data, LT always requires a nav dim
            nav_shape = (1,)
        return nav_shape, sig_shape

    def initialize(self, executor: 'JobExecutor'):
        self._filesize = executor.run_function(self._get_filesize)
        array_meta = executor.run_function(self._read_metadata,
                                           self._path,
                                           use_ds=self._dm_ds_index)
        sig_dims = array_meta['sig_dims']
        self._array_offset = array_meta['offset']
        self._raw_dtype = array_meta['dtype']
        assert self._raw_dtype is not None and not isinstance(self._raw_dtype, int)
        self._array_c_ordered = self._force_c_order or array_meta['c_order']
        nav_shape, sig_shape = self._modify_shape(array_meta['shape'],
                                                  self._array_c_ordered,
                                                  sig_dims=sig_dims)

        # Image count is true number of frames in file (?)
        self._image_count = int(prod(nav_shape))
        if self._nav_shape is not None:
            manual_nav_shape_product = prod(self._nav_shape)
            if manual_nav_shape_product > self._image_count:
                raise DataSetException('Specified nav_shape greater than file nav size')
            elif not self._array_c_ordered and (manual_nav_shape_product != self._image_count):
                raise DataSetException('Manual nav shape with different size '
                                       'not supported for F-ordered DM4')
        else:
            self._nav_shape = nav_shape
        # nav_shape product is either manual nav_shape if supplied or metadata nav_shape (?)
        self._nav_shape_product = int(prod(self._nav_shape))
        sig_size = int(prod(sig_shape))
        if self._sig_shape is not None:
            manual_sig_size = int(prod(self._sig_shape))
            if not self._array_c_ordered and (sig_size != manual_sig_size):
                raise DataSetException('Manual sig shape with different size '
                                       'not supported for F-ordered DM4')
            elif (manual_sig_size * self._nav_shape_product) > (self._image_count * sig_size):
                raise DataSetException('Specified sig_shape and nav size '
                                       'too large for data in file')
        else:
            self._sig_shape = sig_shape

        if not self._array_c_ordered and self._filesize > 2**31:  # warn above 2 GB file
            warnings.warn(
                "This DM dataset is laid out on disk in a way that is inefficient "
                "for LiberTEM to process. Performance may be severely impacted. "
                "To avoid this use a recent version of GMS to acquire data.",
                RuntimeWarning
            )
            if self._io_backend is not None:
                warnings.warn(
                    "An I/O backend was specified which cannot read this "
                    "DM file due to its layout. Using the dataset-specific file "
                    "reader instead.",
                    RuntimeWarning
                )

        # regardless of file order the Dataset shape property is 'standard'
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=sig_dims)
        self._sync_offset_info = self.get_sync_offset_info()
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

    def get_max_io_size(self) -> Optional[int]:
        """
        Override this method to implement a custom maximum I/O size (in bytes)
        """
        if self._array_c_ordered:
            return super().get_max_io_size()
        else:
            # Generate larger tile shapes / depths,
            # limits passes through the file
            return 16 * 2 ** 20

    def get_min_sig_size(self) -> int:
        """
        Prefer small tile sig size and deep tiles
        This will let the UDF(s) choose
        the tileshape without interference
        """
        if self._array_c_ordered:
            return super().get_min_sig_size()
        return 1

    def get_num_partitions(self) -> int:
        if self._array_c_ordered:
            return super().get_num_partitions()
        else:
            # 2 GB partitions or == num cores
            target_part_size = 2048 * 2 ** 20
            ds_bytesize = self.shape.size * np.dtype(self.meta.raw_dtype).itemsize
            return max(self._cores, ds_bytesize // target_part_size, 1)

    def get_partitions(self):
        partition_cls = DMPartition if self._array_c_ordered else DM4PartitionFortran
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
        nav_size = self.shape.nav.size

        itemsize = np.dtype(self.meta.raw_dtype).itemsize
        ds_size_bytes = self.shape.size * itemsize
        tile_depth, tile_sig = tileshape[0], tileshape[1:]
        tile_sig_size_px = prod(tile_sig)
        tile_sig_size_bytes = tile_sig_size_px * itemsize
        # Amount of data we need to traverse to read
        # all data for this tile_sig_shape
        tile_on_disk = tile_sig_size_bytes * nav_size
        if sig_shape == tile_sig:
            # generating frames or partitions
            # nothing to do, in the partition case depth == num frames
            # which will create a huge buffer, but this is inevitable
            return tileshape
        # FIXME Put this onto FortranReader
        # See if MAX_MEMMAP_SIZE + BUFFER_SIZE can be adapted to worker/RAM
        # Approximates the max number of tiles combined in a chunk, clip to ds_size
        # to avoid overestimating the number of tiles and therefore making a shallower depth
        # Do everything in float to avoid div/0 errors
        tiles_in_memmap = min(FortranReader.MAX_MEMMAP_SIZE, ds_size_bytes) / tile_on_disk
        # For the given buffer size, we limit the depth we can read for the
        # max combination of tiles that fit into the memmap
        depth_for_buffer = FortranReader.BUFFER_SIZE / (tile_sig_size_bytes * tiles_in_memmap)
        # Existing tile depth should already be <= part_nframes so clip to this
        depth_for_buffer = max(1, min(tile_depth, int(depth_for_buffer)))
        # A further optimisation would be to half tile_sig and double
        # depth until we hit the partition length, but this could break
        # a UDF which requires a specific minimum tile size or has
        # a maximum depth limit
        return (depth_for_buffer,) + tile_sig

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

    @classmethod
    def validate_tiling_scheme(cls, tiling_scheme: 'TilingScheme'):
        slices = tiling_scheme.slices_array
        shape = tiling_scheme.dataset_shape
        # Group of full columns
        contig_full = (slices[:, 1, :-1] == shape.sig[:-1]).all()
        # Part of a single column, only
        contig_part = (slices[:, 1, 1:] == 1).all()
        if not (contig_full or contig_part):
            raise DataSetException('slices not split in a ravel-able way')


class DM4PartitionFortran(RawPartitionFortran):
    sig_order = 'C'

    @classmethod
    def validate_tiling_scheme(cls, tiling_scheme: 'TilingScheme'):
        slices = tiling_scheme.slices_array
        shape = tiling_scheme.dataset_shape
        # Group of full rows
        contig_full = (slices[:, 1, 1:] == shape.sig[1:]).all()
        # Part of a single row, only
        contig_part = (slices[:, 1, :-1] == 1).all()
        if not (contig_full or contig_part):
            raise DataSetException('slices not split in a ravel-able way')
