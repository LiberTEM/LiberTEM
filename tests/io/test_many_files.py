import os
import pytest
import numpy as np

from libertem.io.dataset.raw import RawFileDataSet, RawFileSet, RawFile

from utils import get_testdata_path

MANYRAW_TESTDATA_PATH = os.path.join(get_testdata_path(), 'many_raw')


def _generate_many_files():
    # Generates 16k 8x8 uint8 frames
    # Combined with the below 2-partition dataset and
    # the two-cpu local_cluster_ctx this is likely to trigger file
    # limits on a system where this is not set to the hard limit globally
    frame = np.arange(64).astype(np.uint8)
    nav_shape = (128, 128)
    os.mkdir(MANYRAW_TESTDATA_PATH)
    for idx in range(np.prod(nav_shape)):
        frame.tofile(os.path.join(MANYRAW_TESTDATA_PATH, f'f{idx:>05d}.raw'))


HAVE_TESTDATA = os.path.exists(MANYRAW_TESTDATA_PATH)
pytestmark = pytest.mark.skipif(not HAVE_TESTDATA, reason="need testdata")  # NOQA


class ManyRawFileDataSetMock(RawFileDataSet):
    """
    A basic multi-file raw dataset that reads every .raw file
    in a folder, assuming one image per frame and all files
    are of an identical size and unordered
    """
    def _get_raw_files(self):
        with os.scandir(self._path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.raw'):
                    yield entry

    def _get_filesize(self):
        return sum(p.stat().st_size for p in self._get_raw_files())

    def _get_fileset(self):
        return RawFileSet([
            RawFile(
                path=p.path,
                start_idx=idx,
                end_idx=idx + 1,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
            for idx, p in enumerate(self._get_raw_files())])

    def get_num_partitions(self):
        return 2


def test_many_files_read(local_cluster_ctx):
    """
    Tests if we can load 8k files per process in a Dask cluster
    on this system. Depending on the system configuration
    this test might be a no-op.
    """
    ds = ManyRawFileDataSetMock(
        path=MANYRAW_TESTDATA_PATH,
        nav_shape=(128, 128),
        sig_shape=(8, 8),
        dtype=np.uint8,
    )
    ds.initialize(local_cluster_ctx.executor)

    sum_a = local_cluster_ctx.create_sum_analysis(ds)
    local_cluster_ctx.run(sum_a)
