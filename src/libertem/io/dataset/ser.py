import os
import logging
import contextlib

from ncempy.io.ser import fileSER

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, File3D, FileSet3D, Partition3D, DataSetException, DataSetMeta,
)

log = logging.getLogger(__name__)


class SERDatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/SERDatasetParams.schema.json",
      "title": "SERDatasetParams",
      "type": "object",
      "properties": {
          "type": {"const": "ser"},
          "path": {"type": "string"},
      },
      "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        return data


class SERFile(File3D):
    def __init__(self, path, emipath=None):
        super().__init__()
        self._path = path
        self._emipath = emipath

    def _get_handle(self):
        return fileSER(self._path, emifile=self._emipath)

    @contextlib.contextmanager
    def get_handle(self):
        with self._get_handle() as f:
            yield f

    @property
    def num_frames(self):
        with self._get_handle() as f1:
            return f1.head['ValidNumberElements']

    @property
    def start_idx(self):
        return 0

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read [`start`, `stop`) images from this file into `out`
        """
        with self._get_handle() as f1:
            num_images = f1.head['ValidNumberElements']
            assert start < num_images
            assert stop <= num_images
            assert stop >= start
            for ii in range(start, stop):
                data0, metadata0 = f1.getDataset(ii)
                if crop_to is not None:
                    # TODO: maybe limit I/O to the cropped region, too?
                    data0 = data0[crop_to.get(sig_only=True)]
                out[ii - start, ...] = data0


class SERFileSet(FileSet3D):
    pass


class SERDataSet(DataSet):
    """
    Read TIA SER files.

    Parameters
    ----------
    path: str
        Path to the .ser file

    emipath: str
        Path to EMI file (currently unused)
    """
    def __init__(self, path, emipath=None):
        super().__init__()
        self._path = path
        self._emipath = emipath
        self._meta = None
        self._filesize = None

    def _get_fileset(self):
        return SERFileSet([
            SERFile(path=self._path, emipath=self._emipath)
        ])

    def initialize(self):
        self._filesize = os.stat(self._path).st_size
        reader = SERFile(path=self._path, emipath=self._emipath)

        with reader.get_handle() as f1:
            if f1.head['ValidNumberElements'] == 0:
                raise DataSetException("no data found in file")

            data, meta_data = f1.getDataset(0)
            dtype = f1._dictDataType[meta_data['DataType']]
            nav_dims = tuple(
                reversed([
                    int(dim['DimensionSize'])
                    for dim in f1.head['Dimensions']
                ])
            )
            shape = nav_dims + tuple(data.shape)
            sig_dims = len(data.shape)
            self._meta = DataSetMeta(
                shape=Shape(shape, sig_dims=sig_dims),
                raw_dtype=dtype,
                iocaps={"FULL_FRAMES"},
            )
        return self

    @classmethod
    def get_msg_converter(cls):
        return SERDatasetParams

    @classmethod
    def detect_params(cls, path):
        if path.lower().endswith(".ser"):
            return {"path": path}
        return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        try:
            with fileSER(self._path, emifile=self._emipath) as f1:
                if f1.head['ValidNumberElements'] == 0:
                    raise DataSetException("no data found in file")
                if f1.head['DataTypeID'] not in (0x4120, 0x4122):
                    raise DataSetException("unknown datatype id: %s" % f1.head['DataTypeID'])
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(self._cores, self._filesize // (512*1024*1024))
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<SERDataSet for %s>" % (self._path,)
