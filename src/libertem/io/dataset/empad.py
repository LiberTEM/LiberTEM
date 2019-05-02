import os
import mmap
from xml.dom import minidom

import numpy as np

from libertem.common import Shape
from .base import DataSet, DataSetException, DataSetMeta, Partition3D
from .raw import RawFile, RawFileSet


EMPAD_DETECTOR_SIZE = (128, 128)
EMPAD_DETECTOR_SIZE_RAW = (130, 128)


def xml_get_text(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


class EMPADFile(RawFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_size = np.product(EMPAD_DETECTOR_SIZE_RAW) * self._meta.raw_dtype.itemsize

    def readinto(self, start, stop, out, crop_to=None):
        """
        readinto copying from mmap, used for cropping off extra data
        (mmap should be preferred for EMPAD data!)
        """
        arr = self.mmap()[start:stop]
        if crop_to is not None:
            arr = arr[(...,) + crop_to.get(sig_only=True)]
        out[:] = arr

    def open(self):
        """
        open file and create memory map
        """
        f = open(self._path, "rb")
        self._file = f
        raw_data = mmap.mmap(
            fileno=f.fileno(),
            length=self.num_frames * self._frame_size,
            offset=self.start_idx * self.num_frames,
            access=mmap.ACCESS_READ,
        )
        self._mmap = np.frombuffer(raw_data, dtype=self._meta.raw_dtype).reshape(
            (self.num_frames,) + EMPAD_DETECTOR_SIZE_RAW
        )[..., :128, :]

    def close(self):
        self._file.close()
        self._file = None
        self._mmap = None


class EMPADFileSet(RawFileSet):
    pass


class EMPADDataSet(DataSet):
    def __init__(self, path, scan_size=None):
        super().__init__()
        self._path = path
        self._scan_size = scan_size and tuple(scan_size) or None
        self._path_raw = None
        self._meta = None

    def _init_from_xml(self, path):
        try:
            dom = minidom.parse(path)
            root = dom.getElementsByTagName("root")[0]
            raw_filename = root.getElementsByTagName("raw_file")[0].getAttribute('filename')
            # because these XML files contain the full path, they are not relocatable.
            # we strip off the path and only use the basename, hoping the .raw file will
            # be in the same directory as the XML file:
            filename = os.path.basename(raw_filename)
            self._path_raw = os.path.join(
                os.path.dirname(path),
                filename
            )
            scan_y = int(xml_get_text(root.getElementsByTagName("pix_y")[0].childNodes))
            scan_x = int(xml_get_text(root.getElementsByTagName("pix_x")[0].childNodes))
            self._scan_size = (scan_y, scan_x)
            # TODO: read more metadata
        except Exception as e:
            raise DataSetException(
                "could not initialize EMPAD file; error: %s" % (
                    str(e))
            )

    def initialize(self):
        lowpath = self._path.lower()
        if lowpath.endswith(".xml"):
            self._init_from_xml(self._path)
        else:
            assert lowpath.endswith(".raw")
            assert self._scan_size is not None
            self._path_raw = self._path

        try:
            self._filesize = os.stat(self._path_raw).st_size
        except OSError as e:
            raise DataSetException("could not open file %s: %s" % (self._path_raw, str(e)))
        self._meta = DataSetMeta(
            shape=Shape(self._scan_size + EMPAD_DETECTOR_SIZE, sig_dims=2),
            raw_dtype=np.dtype("float32"),
            iocaps={"MMAP", "FULL_FRAMES", "FRAME_CROPS"},
        )
        return self

    @classmethod
    def detect_params(cls, path):
        """
        Detect parameters. If an `path` is an xml file, we try to automatically
        set the scan_size, otherwise we can't really detect if this is a EMPAD
        file or something else (maybe from the "trailer" after each frame?)
        """
        try:
            ds = cls(path)
            ds = ds.initialize()
            if not ds.check_valid():
                return False
            return {
                "path": path,
                "scan_size": ds._scan_size,
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
            EMPADFile(
                meta=self._meta,
                path=self._path_raw,
                enable_direct=False,
                enable_mmap=True,
            )
        ])

    def check_valid(self):
        try:
            # check filesize:
            framesize = np.product(EMPAD_DETECTOR_SIZE_RAW)
            num_frames = np.product(self._scan_size)
            expected_filesize = num_frames * framesize * np.dtype("float32").itemsize
            if expected_filesize != self._filesize:
                raise DataSetException("invalid filesize; expected %d, got %d" % (
                    expected_filesize, self._filesize
                ))
            # try to read from the file:
            p = next(self.get_partitions())
            next(p.get_tiles())
            return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB per partition
        res = max(self._cores, self._filesize // (1024*1024*1024))
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<EMPADFileDataSet of %s shape=%s>" % (self.dtype, self.shape)
