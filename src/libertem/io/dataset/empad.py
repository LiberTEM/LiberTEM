import os
from xml.dom import minidom

import numpy as np

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import DataSet, DataSetException, DataSetMeta, BasePartition
from .raw import RawFile, RawFileSet


EMPAD_DETECTOR_SIZE = (128, 128)
EMPAD_DETECTOR_SIZE_RAW = (130, 128)


def xml_get_text(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


def get_params_from_xml(path):
    dom = minidom.parse(path)
    root = dom.getElementsByTagName("root")[0]
    raw_filename = root.getElementsByTagName("raw_file")[0].getAttribute('filename')
    # because these XML files contain the full path, they are not relocatable.
    # we strip off the path and only use the basename, hoping the .raw file will
    # be in the same directory as the XML file:
    filename = os.path.basename(raw_filename)
    path_raw = os.path.join(
        os.path.dirname(path),
        filename
    )

    scan_parameters = [
        elem
        for elem in root.getElementsByTagName("scan_parameters")
        if elem.getAttribute("mode") == "acquire"
    ]

    node_scan_y = scan_parameters[0].getElementsByTagName("scan_resolution_y")[0]
    node_scan_x = scan_parameters[0].getElementsByTagName("scan_resolution_x")[0]

    scan_y = int(xml_get_text(node_scan_y.childNodes))
    scan_x = int(xml_get_text(node_scan_x.childNodes))
    scan_size = (scan_y, scan_x)
    return path_raw, scan_size
    # TODO: read more metadata


class EMPADDatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/EMPADDatasetParams.schema.json",
      "title": "EMPADDatasetParams",
      "type": "object",
      "properties": {
        "type": {"const": "EMPAD"},
        "path": {"type": "string"},
        "scan_size": {
            "type": "array",
            "items": {"type": "number", "minimum": 1},
            "minItems": 2,
            "maxItems": 2
        },
      },
      "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "scan_size"]
        }
        return data


class EMPADFileSet(RawFileSet):
    def __init__(self, *args, **kwargs):
        kwargs.update({
            "frame_footer_bytes": 2*128*4,
        })
        super().__init__(*args, **kwargs)


class EMPADDataSet(DataSet):
    """
    Read data from EMPAD detector. EMPAD data sets consist of two files,
    one .raw and one .xml file. Note that the .xml file contains the file name
    of the .raw file, so if the raw file was renamed at some point, opening using
    the .xml file will fail.

    Parameters
    ----------
    path: str
        Path to either the .xml or the .raw file. If the .xml file given,
        the `scan_size` parameter can be left out

    scan_size: tuple of int
        A tuple (y, x) that specifies the size of the scanned region. It is
        automatically read from the .xml file if you specify one as `path`.
    """
    def __init__(self, path, scan_size=None):
        super().__init__()
        self._path = path
        self._scan_size = scan_size and tuple(scan_size) or None
        self._path_raw = None
        self._meta = None

    def _init_from_xml(self, path):
        try:
            return get_params_from_xml(path)
        except Exception as e:
            raise DataSetException(
                "could not initialize EMPAD file; error: %s" % (
                    str(e))
            )

    def initialize(self, executor):
        lowpath = self._path.lower()
        if lowpath.endswith(".xml"):
            self._path_raw, self._scan_size = executor.run_function(
                self._init_from_xml, self._path
            )
        else:
            assert lowpath.endswith(".raw")
            assert self._scan_size is not None
            self._path_raw = self._path

        try:
            self._filesize = executor.run_function(self._get_filesize)
        except OSError as e:
            raise DataSetException("could not open file %s: %s" % (self._path_raw, str(e)))
        self._meta = DataSetMeta(
            shape=Shape(self._scan_size + EMPAD_DETECTOR_SIZE, sig_dims=2),
            raw_dtype=np.dtype("float32"),
        )
        return self

    def _get_filesize(self):
        return os.stat(self._path_raw).st_size

    @classmethod
    def get_msg_converter(cls):
        return EMPADDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return set(["xml", "raw"])

    @classmethod
    def detect_params(cls, path, executor):
        """
        Detect parameters. If an `path` is an xml file, we try to automatically
        set the scan_size, otherwise we can't really detect if this is a EMPAD
        file or something else (maybe from the "trailer" after each frame?)
        """
        try:
            ds = cls(path)
            ds = ds.initialize(executor)
            if not executor.run_function(ds.check_valid):
                return False
            return {
                "parameters": {
                    "path": path,
                    "scan_size": ds._scan_size,
                },
            }
        except Exception:
            return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def _get_fileset(self):
        num_frames = self._meta.shape.flatten_nav()[0]
        return EMPADFileSet([
            RawFile(
                path=self._path_raw,
                start_idx=0,
                end_idx=num_frames,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
        ])

    def check_valid(self):
        try:
            # check filesize:
            framesize = int(np.prod(EMPAD_DETECTOR_SIZE_RAW, dtype=np.int64))
            num_frames = int(np.prod(self._scan_size))
            expected_filesize = num_frames * framesize * int(np.dtype("float32").itemsize)
            if expected_filesize != self._filesize:
                raise DataSetException("invalid filesize; expected %d, got %d" % (
                    expected_filesize, self._filesize
                ))
            fileset = self._get_fileset()
            with fileset:
                return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_cache_key(self):
        return {
            "path_raw": self._path_raw,
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB per partition
        res = max(self._cores, self._filesize // (1024*1024*1024))
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in BasePartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield BasePartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<EMPADFileDataSet of %s shape=%s>" % (self.dtype, self.shape)
