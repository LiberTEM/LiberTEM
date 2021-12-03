import numpy as np

from libertem.udf import UDF

from utils import _mk_random


class PixelsumBaseUDF(UDF):
    def get_result_buffers(self):
        if self.meta.roi is not None:
            navsize = self.meta.roi.sum()
        else:
            navsize = np.prod(self.meta.dataset_shape.nav)
        return {
            'pixelsum_nav_raw': self.buffer(
                kind="single", dtype="float32", extra_shape=(navsize, )
            ),
            'pixelsum_sig_raw': self.buffer(
                kind="single", dtype="float32", extra_shape=self.meta.dataset_shape.sig
            ),
            'pixelsum_nav': self.buffer(
                kind="nav", dtype="float32"
            ),
            'pixelsum_sig': self.buffer(
                kind="sig", dtype="float32"
            ),
        }

    def get_tiling_preferences(self):
        # Example from SSB UDF: Target tile size depends on dataset shape
        # Make sure we have self.meta available at this stage
        result_size = np.prod(self.meta.dataset_shape.nav)
        target_size = 1024*1024
        return {
            "depth": max(1, target_size // result_size),
            "total_size": target_size,
        }

    def merge(self, dest, src):
        dest.pixelsum_nav_raw[:] += src.pixelsum_nav_raw
        dest.pixelsum_sig_raw[:] += src.pixelsum_sig_raw
        dest.pixelsum_nav[:] += src.pixelsum_nav
        dest.pixelsum_sig[:] += src.pixelsum_sig


class PixelsumPartitionUDF(PixelsumBaseUDF):
    def process_partition(self, partition):
        r = self.results
        r.pixelsum_nav[:] += np.sum(partition, axis=(-1, -2))
        r.pixelsum_sig[:] += np.sum(partition, axis=0)
        r.pixelsum_nav_raw[self.meta.slice.get(nav_only=True)] += np.sum(partition, axis=(-1, -2))
        r.pixelsum_sig_raw[self.meta.slice.get(sig_only=True)] += np.sum(partition, axis=0)


class PixelsumFrameUDF(PixelsumBaseUDF):
    def process_frame(self, frame):
        r = self.results
        r.pixelsum_nav[:] = np.sum(frame)
        r.pixelsum_sig[:] += frame
        r.pixelsum_nav_raw[self.meta.slice.get(nav_only=True)] = np.sum(frame)
        r.pixelsum_sig_raw[self.meta.slice.get(sig_only=True)] += frame


class PixelsumTileUDF(PixelsumBaseUDF):
    def process_tile(self, tile):
        r = self.results
        r.pixelsum_nav[:] += np.sum(tile, axis=(-1, -2))
        r.pixelsum_sig[:] += np.sum(tile, axis=0)
        r.pixelsum_nav_raw[self.meta.slice.get(nav_only=True)] += np.sum(tile, axis=(-1, -2))
        r.pixelsum_sig_raw[self.meta.slice.get(sig_only=True)] += np.sum(tile, axis=0)


def test_partition(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = lt_ctx.load(
        "memory",
        data=data,
        num_partitions=2,
        sig_dims=2
    )

    pixelsum = PixelsumPartitionUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)

    assert np.allclose(res['pixelsum_nav'].data, np.sum(data, axis=(2, 3)))
    assert np.allclose(res['pixelsum_nav_raw'].data.reshape((16, 16)), np.sum(data, axis=(2, 3)))
    assert np.allclose(res['pixelsum_sig'].data, np.sum(data, axis=(0, 1)))
    assert np.allclose(res['pixelsum_sig_raw'].data, np.sum(data, axis=(0, 1)))


def test_frame(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = lt_ctx.load(
        "memory",
        data=data,
        tileshape=(7, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    pixelsum = PixelsumFrameUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)

    assert np.allclose(res['pixelsum_nav'].data, np.sum(data, axis=(2, 3)))
    assert np.allclose(res['pixelsum_nav_raw'].data.reshape((16, 16)), np.sum(data, axis=(2, 3)))
    assert np.allclose(res['pixelsum_sig'].data, np.sum(data, axis=(0, 1)))
    assert np.allclose(res['pixelsum_sig_raw'].data, np.sum(data, axis=(0, 1)))


def test_tile(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = lt_ctx.load(
        "memory",
        data=data,
        tileshape=(7, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    pixelsum = PixelsumTileUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)

    assert np.allclose(res['pixelsum_nav'].data, np.sum(data, axis=(2, 3)))
    assert np.allclose(res['pixelsum_nav_raw'].data.reshape((16, 16)), np.sum(data, axis=(2, 3)))
    assert np.allclose(res['pixelsum_sig'].data, np.sum(data, axis=(0, 1)))
    assert np.allclose(res['pixelsum_sig_raw'].data, np.sum(data, axis=(0, 1)))


def test_roi(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = lt_ctx.load(
        "memory",
        data=data,
        tileshape=(7, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    roi = np.random.choice([True, False], dataset.shape.nav)

    pixelsum = PixelsumTileUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum, roi=roi)

    assert np.allclose(res['pixelsum_nav'].raw_data, np.sum(data[roi], axis=(1, 2)))
    assert np.allclose(res['pixelsum_nav_raw'].data, np.sum(data[roi], axis=(1, 2)))
    assert np.allclose(res['pixelsum_sig'].raw_data, np.sum(data[roi], axis=0))
    assert np.allclose(res['pixelsum_sig_raw'].data, np.sum(data[roi], axis=0))
