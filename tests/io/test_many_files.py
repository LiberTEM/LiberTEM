import os
import pytest
import numpy as np

from libertem.io.dataset.raw import RawFileDataSet, RawFileSet, RawFile


RUNNER_OS = os.environ.get("RUNNER_OS", "").lower().strip()
RUNNER_ARCH = os.environ.get("RUNNER_ARCH", "").lower().strip()

try:
    import resource
    _, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    pytestmark = pytest.mark.skipif(hard_lim < 2 ** 15, reason="hard file limit is too low")  # NOQA
except ModuleNotFoundError:
    # Not available on Windows
    pass


@pytest.fixture(scope='session')
def generate_many_files(tmpdir_factory):
    # Generates 16k 8x8 uint8 frames
    # Combined with the below 2-partition dataset and
    # the two-cpu local_cluster_ctx this is likely to trigger file
    # limits on a system where this is not set to the hard limit globally
    datadir = tmpdir_factory.mktemp('many_raw_files')
    frame = np.arange(64).astype(np.uint8)
    nav_shape = (128, 100)
    for idx in range(np.prod(nav_shape)):
        frame.tofile(os.path.join(datadir, f'f{idx:>05d}.raw'))
    yield datadir


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


@pytest.mark.skipif(
    RUNNER_OS.startswith('macos') and RUNNER_ARCH.startswith("arm"),
    reason="Hard limit not compatible on macos-14 runner"
)
def test_many_files_read(local_cluster_ctx, generate_many_files):
    """
    Tests if we can load 8k files per process in a Dask cluster
    on this system. Depending on the system configuration
    this test might be a no-op.
    """
    ds = ManyRawFileDataSetMock(
        path=generate_many_files,
        nav_shape=(128, 100),
        sig_shape=(8, 8),
        dtype=np.uint8,
    )
    ds.initialize(local_cluster_ctx.executor)

    sum_a = local_cluster_ctx.create_sum_analysis(ds)
    local_cluster_ctx.run(sum_a)
