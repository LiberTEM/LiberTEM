import h5py
import numpy as np
from PIL import Image

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
            for k in self.get_result_keys():
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
            "description": "numpy format (.npz)",
        }

    def _get_result_dict(self):
        return {
            k: np.array(self._result_set[k])
            for k in self.get_result_keys()
        }

    def serialize_to_buffer(self, buf):
        np.savez(buf, **self._get_result_dict())

    def get_content_type(self):
        return "application/octet-stream"

    def get_filename(self):
        return "results.npz"  # FIXME: naming


class NPZCompressedResultFormat(NPZResultFormat):
    @classmethod
    def format_info(cls):
        return {
            "id": "NPZ_COMPRESSED",
            "description": "numpy format, compressed (.npz)",
        }

    def serialize_to_buffer(self, buf):
        np.savez_compressed(buf, **self._get_result_dict())


class TiffResultFormat(ResultFormat):
    @classmethod
    def format_info(cls):
        return {
            "id": "TIFF",
            "description": "Multi-page 32bit float TIFF (.tif)",
        }

    def get_channel_images(self):
        for k in self.get_result_keys():
            result = np.array(self._result_set[k]).astype(np.float32)
            yield Image.fromarray(result)

    def serialize_to_buffer(self, buf):
        images = self.get_channel_images()
        first_image = next(images)
        first_image.save(buf, format="TIFF", save_all=True, append_images=images)

    def get_content_type(self):
        return "image/tiff"

    def get_filename(self):
        return "results.tif"  # FIXME: naming


class RawResultFormat(ResultFormat):
    @classmethod
    def format_info(cls):
        return {
            "id": "RAW",
            "description": "Raw binary, as-is (.bin)",
        }

    def _get_result_arr(self):
        return np.stack([
            np.array(self._result_set[k])
            for k in self.get_result_keys()
        ])

    def serialize_to_buffer(self, buf):
        buf.write(self._get_result_arr().tobytes())

    def get_content_type(self):
        return "application/octet-stream"

    def get_filename(self):
        arr = self._get_result_arr()
        return "results_{}_{}.bin".format(arr.dtype, "-".join(str(i) for i in arr.shape))
