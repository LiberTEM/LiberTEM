import os
import logging

from ncempy.io.mrc import fileMRC

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import DataSet, FileSet, BasePartition, DataSetMeta, LocalFile

log = logging.getLogger(__name__)


class MRCDatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/MRCDatasetParams.schema.json",
      "title": "MRCDatasetParams",
      "type": "object",
      "properties": {
          "type": {"const": "MRC"},
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


class MRCFile(LocalFile):
    def open(self):
        self._file = fileMRC(self._path)
        self._mmap = self._file.getMemmap()
        self._raw_mmap = self._mmap

    def close(self):
        self._file = None
        self._mmap = None
        self._raw_mmap = None

    def fileno(self):
        return None


class MRCFileSet(FileSet):
    pass


class MRCDataSet(DataSet):
    """
    Read MRC files.

    Parameters
    ----------
    path: str
        Path to the .mrc file
    """
    def __init__(self, path, sig_shape=None):
        super().__init__()
        self._path = path
        self._meta = None
        self._filesize = None
        self._num_frames = None
        self._sig_shape = sig_shape

    def _do_initialize(self):
        self._filesize = os.stat(self._path).st_size
        f = fileMRC(self._path)
        data = f.getMemmap()
        shape = data.shape
        dtype = data.dtype

        # FIXME: make sig_shape overridable (pending Anand's PR)
        sig_shape = tuple(
            i
            for i in f.gridSize
            if i != 1
        )

        self._meta = DataSetMeta(
            shape=Shape(shape, sig_dims=len(sig_shape)),
            raw_dtype=dtype,
        )
        self._num_frames = self._meta.shape.nav.size
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @classmethod
    def get_msg_converter(cls):
        return MRCDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return set(["mrc"])

    @classmethod
    def detect_params(cls, path, executor):
        if path.lower().endswith(".mrc"):
            # FIXME: sig_shape, nav_shape
            return {
                "parameters": {
                    "path": path
                }
            }
        return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        return True  # anything to check?

    def get_cache_key(self):
        return {
            "path": self._path,
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(self._cores, self._filesize // (64*1024*1024))
        return res

    def _get_fileset(self):
        assert self._num_frames is not None
        return MRCFileSet([
            MRCFile(
                path=self._path,
                start_idx=0,
                end_idx=self._num_frames,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
        ])

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in MRCPartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield MRCPartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<MRCDataSet for %s>" % (self._path,)


class MRCPartition(BasePartition):
    pass
