import math
import itertools
import pathlib
import contextlib
import numpy as np
import pytest

from libertem.common.shape import Shape
from libertem.udf.raw import PickUDF
from libertem.udf.sum import SumUDF
from libertem.io.dataset.raw_group import RawFileGroupDataSet
from libertem.io.dataset.base import BufferedBackend, MMapBackend, DirectBackend
from libertem.io.dataset.base import Negotiator, DataSetException

from utils import _mk_random


def random_bytes(n):
    return np.random.randint(0, 255, size=(n,)).astype(np.uint8).tobytes()


def group_frames(frame_iterator, n, file_header, frame_header, frame_footer):
    frames = []
    for _ in range(n):
        try:
            frame = (random_bytes(frame_header)
                    + next(frame_iterator).tobytes()
                    + random_bytes(frame_footer))
            frames.append(frame)
        except StopIteration:
            pass
    return len(frames), random_bytes(file_header) + b''.join(frames)


def raw_group_ds(tmpdir_factory,
                 frames_per_file, file_header,
                 frame_header, frame_footer,
                 nav_shape=(16, 16),
                 sig_shape=(64, 64),
                 dtype=np.float32):
    if isinstance(frames_per_file, int):
        if frames_per_file == -1:
            frames_per_file = math.prod(nav_shape)
        frames_per_file = (frames_per_file,)
    frames_per_file = itertools.cycle(frames_per_file)

    datadir = pathlib.Path(tmpdir_factory.mktemp('data'))
    raw_data = _mk_random(size=(math.prod(nav_shape),) + sig_shape, dtype=dtype)

    paths = []
    frame_iterator = iter(raw_data)

    frames = 0
    while frames < raw_data.shape[0]:
        _frames, fbytes = group_frames(frame_iterator,
                                       next(frames_per_file),
                                       file_header,
                                       frame_header,
                                       frame_footer)
        frames += _frames
        filename = datadir / f'{len(paths):>04d}.raw'
        with filename.open('wb') as fp:
            fp.write(fbytes)
        paths.append(filename)

    return raw_data.reshape(nav_shape + sig_shape), paths


@contextlib.contextmanager
def get_dataset(tmpdir_factory, lt_ctx,
                frames_per_file=-1,
                file_header=0,
                frame_header=0,
                frame_footer=0,
                nav_shape=(16, 16),
                sig_shape=(64, 64),
                dtype=np.float32,
                backend=None):
    # Files per frame -1 corresponds to one raw file containing all frames
    raw_data, paths = raw_group_ds(tmpdir_factory,
                                   frames_per_file=frames_per_file,
                                   file_header=file_header,
                                   frame_header=frame_header,
                                   frame_footer=frame_footer,
                                   dtype=dtype,
                                   nav_shape=nav_shape,
                                   sig_shape=sig_shape)
    shape = Shape(raw_data.shape, 2)
    ds = RawFileGroupDataSet(paths=paths,
                             nav_shape=shape.nav,
                             sig_shape=shape.sig,
                             dtype=raw_data.dtype,
                             file_header=file_header,
                             frame_header=frame_header,
                             frame_footer=frame_footer,
                             io_backend=backend)
    ds.initialize(lt_ctx.executor)
    yield ds, raw_data
    _ = [f.unlink() for f in paths]


@pytest.mark.parametrize("file_header", (0, 4))
@pytest.mark.parametrize("frame_header", (0, 4))
@pytest.mark.parametrize("frame_footer", (0, 4))
@pytest.mark.parametrize("frames_per_file", (-1, 1, 3))
@pytest.mark.parametrize("dtype", (np.float32, np.int16))
@pytest.mark.parametrize("backend_cls", (BufferedBackend, MMapBackend, DirectBackend))
def test_w_copy(tmpdir_factory, lt_ctx,
                file_header, frame_header, frame_footer,
                frames_per_file, backend_cls, dtype):
    with get_dataset(tmpdir_factory, lt_ctx,
                     frames_per_file=frames_per_file,
                     file_header=file_header,
                     frame_header=frame_header,
                     frame_footer=frame_footer,
                     backend=backend_cls(),
                     dtype=dtype) as (ds, raw_data):
        udf = PickUDF()
        # Using an ROI forces copy-based reading
        mask = np.zeros(ds.meta.shape.nav, dtype=bool)
        h, w = mask.shape
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        mask[y, x] = True

        result = lt_ctx.run_udf(dataset=ds, udf=udf, roi=mask)
        frame = result['intensity'].data.squeeze(axis=0)
        assert np.allclose(frame, raw_data[y, x])


@pytest.mark.parametrize("file_header", (0, 4))
@pytest.mark.parametrize("frame_header", (0, 4))
@pytest.mark.parametrize("frame_footer", (0, 4))
@pytest.mark.parametrize("nav_shape", ((16, 16),))
@pytest.mark.parametrize("frames_per_file", (16,))
@pytest.mark.parametrize("backend_cls", (MMapBackend,))
def test_straight(tmpdir_factory, lt_ctx,
                  file_header, frame_header, frame_footer,
                  frames_per_file, nav_shape, backend_cls):
    with get_dataset(tmpdir_factory, lt_ctx,
                     frames_per_file=frames_per_file,
                     file_header=file_header,
                     frame_header=frame_header,
                     frame_footer=frame_footer,
                     nav_shape=nav_shape,
                     backend=backend_cls()) as (ds, raw_data):
        # Must hack num_partitions to force straight reading
        # due to default tileshape.depth > frames_per_file
        ds.get_num_partitions = lambda: ds.meta.shape.nav[0]
        udf = SumUDF()
        # Check we will actually read straight on the first partition!!
        read_dtype = np.result_type(udf.get_preferred_input_dtype(), ds.dtype)
        partition = next(ds.get_partitions())
        tiling_scheme = Negotiator().get_scheme(
            udfs=[udf],
            approx_partition_shape=partition.shape,
            dataset=ds,
            read_dtype=read_dtype,
        )
        need_copy = ds.get_io_backend().get_impl().need_copy(
            decoder=ds.get_decoder(),
            roi=None,
            native_dtype=ds.meta.raw_dtype,
            read_dtype=read_dtype,
            sync_offset=ds._sync_offset,
            tiling_scheme=tiling_scheme,
            fileset=ds._get_fileset(),
        )
        assert not need_copy
        result = lt_ctx.run_udf(dataset=ds, udf=udf)
        sum_frame = result['intensity'].data
        assert np.allclose(sum_frame, raw_data.sum(axis=(0, 1)))


@pytest.mark.parametrize("frame_header_footer", ((4, 7), (7, 4)))
@pytest.mark.parametrize("backend_cls", (MMapBackend,))
@pytest.mark.parametrize("dtype", (np.float32, np.int16))
def test_mmap_nonitemsize_failure(tmpdir_factory, lt_ctx,
                                  frame_header_footer,
                                  dtype, backend_cls):
    # Tests that we raise if using MMapBackend with a frame_header
    # or footer which is not a multiple of dtype.itemsize
    with pytest.raises(DataSetException):
        with get_dataset(tmpdir_factory, lt_ctx,
                         frames_per_file=3,
                         frame_header=frame_header_footer[0],
                         frame_footer=frame_header_footer[1],
                         backend=backend_cls(),
                         dtype=dtype) as _:
            pass


@pytest.mark.parametrize("frame_header", (7,))
@pytest.mark.parametrize("backend_cls", (BufferedBackend,))
@pytest.mark.parametrize("dtype", (np.float32, np.int16))
def test_buffered_nonitemsize(tmpdir_factory, lt_ctx,
                              frame_header, backend_cls, dtype):
    # Shows that BufferedBackend does support frame_header
    # which is not a multiple of dtype.itemsize
    with get_dataset(tmpdir_factory, lt_ctx,
                     frames_per_file=3,
                     frame_header=frame_header,
                     backend=backend_cls(),
                     dtype=dtype) as (ds, raw_data):
        udf = SumUDF()
        result = lt_ctx.run_udf(dataset=ds, udf=udf)
        sum_frame = result['intensity'].data
        assert np.allclose(sum_frame, raw_data.sum(axis=(0, 1)))
