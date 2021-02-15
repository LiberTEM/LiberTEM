import os
import warnings
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

    nav_y = int(xml_get_text(node_scan_y.childNodes))
    nav_x = int(xml_get_text(node_scan_x.childNodes))
    nav_shape = (nav_y, nav_x)
    return path_raw, nav_shape
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
      },
      "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
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
        the `nav_shape` parameter can be left out

    nav_shape: tuple of int, optional
        A tuple (y, x) that specifies the size of the scanned region. It is
        automatically read from the .xml file if you specify one as `path`.

    sig_shape: tuple of int, optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    """
    def __init__(self, path, scan_size=None, nav_shape=None,
                 sig_shape=None, sync_offset=0, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._path = path
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
        nav_shape_from_XML = None
        lowpath = self._path.lower()
        if lowpath.endswith(".xml"):
            self._path_raw, nav_shape_from_XML = executor.run_function(
                self._init_from_xml, self._path
            )
        else:
            if not lowpath.endswith(".raw"):
                raise DataSetException("path should either be .xml or .raw")
            if self._nav_shape is None:
                raise DataSetException("need to set or detect nav_shape!")
            self._path_raw = self._path

        try:
            self._filesize = executor.run_function(self._get_filesize)
        except OSError as e:
            raise DataSetException("could not open file %s: %s" % (self._path_raw, str(e)))
        self._image_count = int(
            self._filesize / (
                int(np.dtype("float32").itemsize) * int(
                    np.prod(EMPAD_DETECTOR_SIZE_RAW, dtype=np.int64)
                )
            )
        )
        if self._nav_shape is None and nav_shape_from_XML is not None:
            self._nav_shape = nav_shape_from_XML
        elif self._nav_shape is None and nav_shape_from_XML is None:
            raise ValueError(
                    "either nav_shape needs to be passed, or path needs to point to the .xml file"
                )
        self._nav_shape_product = int(np.prod(self._nav_shape))
        if nav_shape_from_XML:
            self._image_count = int(np.prod(nav_shape_from_XML))
        if self._sig_shape is None:
            self._sig_shape = EMPAD_DETECTOR_SIZE
        elif int(np.prod(self._sig_shape)) != int(np.prod(EMPAD_DETECTOR_SIZE)):
            raise DataSetException(
                "sig_shape must be of size: %s" % int(np.prod(EMPAD_DETECTOR_SIZE))
            )
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape)),
            raw_dtype=np.dtype("float32"),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
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
        set the nav_shape, otherwise we can't really detect if this is a EMPAD
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
                    "nav_shape": ds._nav_shape,
                    "sig_shape": ds._sig_shape,
                },
                "info": {
                    "image_count": ds._image_count,
                    "native_sig_shape": ds._sig_shape,
                }
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
        return EMPADFileSet([
            RawFile(
                path=self._path_raw,
                start_idx=0,
                end_idx=self._image_count,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
        ])

    def check_valid(self):
        try:
            fileset = self._get_fileset()
            with fileset:
                return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_cache_key(self):
        return {
            "path_raw": self._path_raw,
            "shape": tuple(self.shape),
            "sync_offset": self._sync_offset,
        }

    def get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB per partition
        res = max(self._cores, self._filesize // (1024*1024*1024))
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield BasePartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<EMPADFileDataSet of %s shape=%s>" % (self.dtype, self.shape)
