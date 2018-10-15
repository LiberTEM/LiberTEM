import re
import io
import os
import glob

import numpy as np

from libertem.common.slice import Slice
from .base import DataSet, Partition, DataTile, DataSetException


class MIBFile(object):
    def __init__(self, path):
        self.path = path
        self._fields = {}

    def _get_np_dtype(self, dtype):
        dtype = dtype.lower()
        assert dtype[0] == "u"
        num_bytes = int(dtype[1:]) // 8
        return ">u%d" % num_bytes

    def read_header(self):
        with io.open(file=self.path, mode="r", encoding="ascii") as f:
            header = f.read(100)
            filesize = os.fstat(f.fileno()).st_size
        parts = header.split(",")
        image_size = (int(parts[5]), int(parts[4]))
        header_size_bytes = int(parts[2])
        bytes_per_pixel = int(parts[6][1:]) // 8
        num_images = filesize // (
            image_size[0] * image_size[1] * bytes_per_pixel + header_size_bytes
        )
        self._fields = {
            'header_size_bytes': header_size_bytes,
            'dtype': self._get_np_dtype(parts[6]),
            'bytes_per_pixel': bytes_per_pixel,
            'image_size': image_size,
            'sequence_first_image': int(parts[1]),
            'filesize': filesize,
            'num_images': num_images,
        }
        return self._fields

    @property
    def fields(self):
        if not self._fields:
            self.read_header()
        return self._fields

    def _frames(self, num, offset):
        """
        read frames as views into the memmapped file

        Parameters
        ----------

        num : int
            number of frames to read
        offset : int
            index of first frame to read (number of frames to skip)
        """
        bpp = self.fields['bytes_per_pixel']
        hsize = self.fields['header_size_bytes']
        size_px = self.fields['image_size'][0] * self.fields['image_size'][1]
        size = size_px * bpp  # bytes
        imagesize_incl_header = size + hsize  # bytes
        mapped = np.memmap(self.path, dtype=self.fields['dtype'], mode='r',
                           offset=offset * imagesize_incl_header)
        idx = 0
        while idx < num:
            start = idx * (imagesize_incl_header // bpp) + hsize // bpp
            end = start + size_px
            yield idx, mapped[start:end]
            idx += 1

    def read_frames(self, num, offset, out):
        """
        Read a number of frames into an existing buffer, skipping the headers

        Parameters
        ----------

        num : int
            number of frames to read
        offset : int
            index of first frame to read
        out : buffer
            output buffer that should fit `num` frames
        """
        imagesize = self.fields['image_size']
        out_reshaped = out.reshape(num, imagesize[0] * imagesize[1])

        for idx, frame in self._frames(num=num, offset=offset):
            out_reshaped[idx] = frame
        return out


class MIBDataSet(DataSet):
    def __init__(self, path, tileshape, scan_size):
        self._path = path
        self._tileshape = tileshape
        self._scan_size = tuple(scan_size)

    def _files(self):
        path, ext = os.path.splitext(self._path)
        if ext == '.mib':
            pattern = "%s*.mib" % (
                re.sub(r'[0-9]+$', '', path)
            )
        elif ext == '.hdr':
            pattern = "%s*.mib" % path
        else:
            raise DataSetException("unknown extension")

        for path in glob.glob(pattern):
            yield MIBFile(path)

    def _files_sorted(self):
        return sorted(self._files(), key=lambda f: f.fields['sequence_first_image'])

    def _first_file(self):
        return next(iter(self._files_sorted()))

    @property
    def dtype(self):
        first_file = self._first_file()
        return first_file.fields['dtype']

    @property
    def shape(self):
        first_file = self._first_file()
        return self._scan_size + first_file.fields['image_size']

    def check_valid(self):
        try:
            s = self._scan_size
            num_images = sum(f.fields['num_images'] for f in self._files())
            if s[0] * s[1] != num_images:
                raise DataSetException(
                    "scan_size (%r) does not match number of images (%d)" % (
                        s, num_images
                    )
                )
            if tuple(self._tileshape[2:]) != self.shape[2:]:
                raise DataSetException(
                    "MIB only supports tileshapes that match whole frames, %r != %r" % (
                        self._tileshape[2:], self.shape[2:]
                    )
                )
            if self._tileshape[0] != 1:
                raise DataSetException(
                    "MIB only supports tileshapes that don't cross rows"
                )
            # FIXME: this should not generally be required!
            """
            for f in self._files():
                if f.fields['num_images'] % self._scan_size[0] != 0:
                    raise DataSetException(
                        "only supporting rectangular shapes per file for now"
                    )
            """
        except (IOError, OSError, KeyError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        """
        we keep it simple: one MIB file == one partition
        """

        ds_slice = Slice(origin=(0, 0, 0, 0), shape=self.shape)
        for f in self._files_sorted():
            idx = f.fields['sequence_first_image'] - 1
            length = f.fields['num_images']

            pslice = ds_slice.subslice_from_offset(offset=idx, length=length)

            yield MIBPartition(
                tileshape=self._tileshape,
                dataset=self,
                partfile=f,
                dtype=self.dtype,
                partition_slice=pslice,
            )

    def __repr__(self):
        return "<MIBDataSet of %s shape=%s>" % (self.dtype, self.shape)


class MIBPartition(Partition):
    def __init__(self, tileshape, partfile, *args, **kwargs):
        self.tileshape = tileshape
        self.partfile = partfile
        super().__init__(*args, **kwargs)
        assert all(s > 0 for s in self.shape), "invalid shape (%r)" % (self.shape,)

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape[2:] != self.dataset.shape[2:]:
                raise DataSetException("MIBDataSet only supports whole-frame crops for now")
        stackheight = self.tileshape[1]

        data = np.ndarray(self.tileshape, dtype=self.dtype)
        num_tiles = self.partfile.fields['num_images'] // stackheight

        for t in range(num_tiles):
            tile_slice = self.slice.subslice_from_offset(offset=t * stackheight,
                                                         length=stackheight)
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            self.partfile.read_frames(num=stackheight, offset=t * stackheight, out=data)
            yield DataTile(data=data, tile_slice=tile_slice)

    def get_locations(self):
        return "127.0.1.1"  # FIXME
