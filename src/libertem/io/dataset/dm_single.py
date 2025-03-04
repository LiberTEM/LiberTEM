import os
import warnings
import typing
from typing import Any, Optional
import logging

import ncempy.io.dm
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape
from .base import BasePartition, DataSetException, DataSetMeta, File, IOBackend, DataSet
from .dm import DMDataSet, SingleDMDatasetParams, DMFileSet

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from numpy import typing as nt
    from libertem.common.executor import JobExecutor


class SingleDMDataSet(DMDataSet):
    """
    Reader for a single DM3/DM4 file. Handles 4D-STEM, 3D-Spectrum Images,
    and TEM image stacks stored in a single-file format. Where possible
    the structure will be inferred from the file metadata.

    .. versionadded:: 0.11.0

    Note
    ----
    Single-file DM data can be stored on disk using either normal C-ordering,
    which is an option in recent versions of GMS, or an alternative F/C-hybrid
    ordering depending on the imaging mode and dimensionality. The reading of
    F/C-hybrid files is currently not supported for performance reasons.

    The DataSet will try to infer the ordering from the file metadata and
    read accordingly. If the file uses the older hybrid F/C-ordering
    :code:`(flat_sig, flat_nav)` then the dataset will raise an exception
    unless the `force_c_order` argument. is set to true.

    A converter for F/C-hybrid files is provided as
    :meth:`~libertem.contrib.convert_transposed.convert_dm4_transposed`.

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

    nav_shape : Tuple[int, ...], optional
        Over-ride the nav_shape provided by the file metadata.
        This can be used to adjust the total
        number of frames.

    sig_shape: Tuple[int, ...], optional
        Over-ride the sig_shape provided by the file metadata.
        Data are read sequentially in all cases, therefore this
        is typically only interesting if the total number of
        sig pixels remains constant.

    sync_offset: int, optional, by default 0
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    io_backend: IOBackend, optional
        A specific IOBackend implementation to over-ride the
        platform default.

    force_c_order: bool, optional, by default False
        Force the data to be interpreted as a C-ordered
        array regardless of the tag information. This will lead
        to incorrect results on an hybrid C/F-ordered file.

    dataset_index: int, optional
        In the case of a multi-dataset DM file this can
        be used to open a specific dataset index. Note that
        the datasets in a DM-file often begin with a thumbnail
        which occupies the 0 dataset index. If not provided the
        first compatible dataset found in the file is used.

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.
    """
    def __init__(
        self,
        path: os.PathLike,
        nav_shape: Optional[tuple[int, ...]] = None,
        sig_shape: Optional[tuple[int, ...]] = None,
        sync_offset: int = 0,
        io_backend: Optional[IOBackend] = None,
        force_c_order: bool = False,
        dataset_index: Optional[int] = None,
        num_partitions: Optional[int] = None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        self._filesize = None

        self._path = path
        self._nav_shape = tuple(nav_shape) if nav_shape else None
        self._sig_shape = tuple(sig_shape) if sig_shape else None
        self._sync_offset = sync_offset
        self._force_c_order = force_c_order
        self._dm_ds_index = dataset_index

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
                return False
            sync_offset = 0
            nav_shape, sig_shape = cls._modify_shape(array_meta['shape'],
                                                    sig_dims=sig_dims)
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
    def _tags_to_nest(tags: dict[str, Any]):
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
    def _modify_shape(shape: tuple[int, ...], sig_dims: int = 2):
        # The shape reversal to read in C-ordering applies to DM4/STEM files
        # saved in the new style as well as DM3, 3D image stacks saved
        # in older versions of GMS. Must check whether newer image stacks
        # are saved in C-ordering as well (despite the metadata order)
        shape = tuple(reversed(shape))
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
        array_c_ordered = self._force_c_order or array_meta['c_order']

        if not array_c_ordered:
            raise DataSetException('Cannot identify DM file as C-ordered from metadata'
                                   'use force_c_order=True to force behaviour.')

        nav_shape, sig_shape = self._modify_shape(array_meta['shape'],
                                                  sig_dims=sig_dims)

        # Image count is true number of frames in file (?)
        self._image_count = int(prod(nav_shape))
        if self._nav_shape is not None:
            manual_nav_shape_product = prod(self._nav_shape)
            if manual_nav_shape_product > self._image_count:
                raise DataSetException('Specified nav_shape greater than file nav size')
        else:
            self._nav_shape = nav_shape
        # nav_shape product is either manual nav_shape if supplied or metadata nav_shape (?)
        self._nav_shape_product = int(prod(self._nav_shape))
        sig_size = int(prod(sig_shape))
        if self._sig_shape is not None:
            manual_sig_size = int(prod(self._sig_shape))
            if (manual_sig_size * self._nav_shape_product) > (self._image_count * sig_size):
                raise DataSetException('Specified sig_shape and nav size '
                                       'too large for data in file')
        else:
            self._sig_shape = sig_shape

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

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield DMPartition(
                meta=self.meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )


class DMFile(File):
    ...


class DMPartition(BasePartition):
    ...
