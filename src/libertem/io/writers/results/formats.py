import h5py
import numpy as np

from .base import ResultFormat


class HDF5ResultFormat(ResultFormat):
    @classmethod
    def format_info(cls):
        return {
            "id": "HDF5",
            "description": "HDF5 container (.h5)",
        }

    def serialize_to_buffer(self, buf):
        with h5py.File(buf, 'w') as f:
            for k in self._result_set.keys():
                f[k] = self._result_set[k]
            # FIXME: add "metadata", for example, what analysis was run that
            # resulted in this file

    def get_content_type(self):
        return "application/x-hdf5"

    def get_filename(self):
        return "results.h5"  # FIXME: more specific name


class NPZResultFormat(ResultFormat):
    @classmethod
    def format_info(cls):
        return {
            "id": "NPZ",
            "description": "Numpy format (.npz)",
        }

    def _get_result_dict(self):
        return {
            k: self._result_set[k]
            for k in self._result_set.keys()
        }

    def serialize_to_buffer(self, buf):
        np.savez(buf, self._get_result_dict())

    def get_content_type(self):
        return "application/octet-stream"

    def get_filename(self):
        return "results.npy"  # FIXME: naming


class NPZCompressedResultFormat(ResultFormat):
    @classmethod
    def format_info(cls):
        return {
            "id": "NPZ_COMPRESSED",
            "description": "Numpy format, compressed (.npz)",
        }

    def serialize_to_buffer(self, buf):
        np.savez_compressed(buf, self._get_result_dict())
