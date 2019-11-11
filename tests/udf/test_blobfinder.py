import functools

import numpy as np
import pytest

import libertem.udf.blobfinder as blobfinder
import libertem.analysis.gridmatching as grm
import libertem.masks as m
from libertem.utils.generate import cbed_frame

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_refinement():
    data = np.array([
        (0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (2, 3, 0, 0, 0, 0),
        (0, 2, 0, 0, 0, -10)
    ])

    assert np.allclose(blobfinder.refine_center(center=(1, 1), r=1, corrmap=data), (1, 1))
    assert np.allclose(blobfinder.refine_center(center=(2, 2), r=1, corrmap=data), (1, 1))
    assert np.allclose(blobfinder.refine_center(center=(1, 4), r=1, corrmap=data), (0.5, 4.5))

    y, x = (4, 1)
    ry, rx = blobfinder.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry > y) and (ry < (y + 1))
    assert (rx < x) and (rx > (x - 1))

    y, x = (4, 4)
    ry, rx = blobfinder.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry < y) and (ry > (y - 1))
    assert (rx < x) and (rx > (x - 1))


def test_smoke(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)
    match_pattern = blobfinder.RadialGradient(radius=4)
    blobfinder.run_blobfinder(
        ctx=lt_ctx, dataset=dataset, num_peaks=1, match_pattern=match_pattern
    )


def test_crop_disks_from_frame():
    match_pattern = blobfinder.RadialGradient(radius=2, search=2)
    peaks = [
        [0, 0],
        [2, 2],
        [5, 5],
    ]
    frame = _mk_random(size=(6, 6), dtype="float32")
    crop_disks = list(blobfinder.crop_disks_from_frame(
        peaks,
        frame,
        match_pattern
    ))

    #
    # how is the region around the peak cropped? like this (x denotes the peak position),
    # this is an example for radius 2, padding 0 -> crop_size = 4
    #
    # ---------
    # | | | | |
    # |-|-|-|-|
    # | | | | |
    # |-|-|-|-|
    # | | |x| |
    # |-|-|-|-|
    # | | | | |
    # ---------

    # first peak: top-leftmost; only the bottom right part of the crop_buf should be filled:
    assert crop_disks[0][1] == (slice(2, 4), slice(2, 4))

    # second peak: the whole crop area fits into the frame -> use full crop_buf
    assert crop_disks[1][1] == (slice(0, 4), slice(0, 4))

    # third peak: bottom-rightmost; almost-symmetric to first case
    assert crop_disks[2][1] == (slice(0, 3), slice(0, 3))


def test_run_refine_fastmatch(lt_ctx):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)
    matcher = grm.Matcher()

    template = m.radial_gradient(
        centerX=radius+1,
        centerY=radius+1,
        imageSizeX=2*radius+2,
        imageSizeY=2*radius+2,
        radius=radius
    )

    match_patterns = [
        blobfinder.RadialGradient(radius=radius),
        blobfinder.Circular(radius=radius),
        blobfinder.BackgroundSubtraction(radius=radius),
        blobfinder.RadialGradientBackgroundSubtraction(radius=radius),
        blobfinder.UserTemplate(template=template)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        (res, real_indices) = blobfinder.run_refine(
            ctx=lt_ctx,
            dataset=dataset,
            zero=zero + np.random.uniform(-1, 1, size=2),
            a=a + np.random.uniform(-1, 1, size=2),
            b=b + np.random.uniform(-1, 1, size=2),
            matcher=matcher,
            match_pattern=match_pattern
        )
        print(peaks - grm.calc_coords(
            res['zero'].data[0],
            res['a'].data[0],
            res['b'].data[0],
            indices)
        )

        assert np.allclose(res['zero'].data[0], zero, atol=0.5)
        assert np.allclose(res['a'].data[0], a, atol=0.2)
        assert np.allclose(res['b'].data[0], b, atol=0.2)


def test_run_refine_affinematch(lt_ctx):
    for i in range(1):
        try:
            shape = np.array([128, 128])

            zero = shape / 2 + np.random.uniform(-1, 1, size=2)
            a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
            b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)

            indices = np.mgrid[-2:3, -2:3]
            indices = np.concatenate(indices.T)

            radius = 10

            data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

            dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                                    num_partitions=1, sig_dims=2)

            matcher = grm.Matcher()
            match_pattern = blobfinder.RadialGradient(radius=radius)

            affine_indices = peaks - zero

            for j in range(5):
                zzero = zero + np.random.uniform(-1, 1, size=2)
                aa = np.array([1, 0]) + np.random.uniform(-0.05, 0.05, size=2)
                bb = np.array([0, 1]) + np.random.uniform(-0.05, 0.05, size=2)

                (res, real_indices) = blobfinder.run_refine(
                    ctx=lt_ctx,
                    dataset=dataset,
                    zero=zzero,
                    a=aa,
                    b=bb,
                    indices=affine_indices,
                    matcher=matcher,
                    match_pattern=match_pattern,
                    match='affine'
                )

                assert np.allclose(res['zero'].data[0], zero, atol=0.5)
                assert np.allclose(res['a'].data[0], [1, 0], atol=0.05)
                assert np.allclose(res['b'].data[0], [0, 1], atol=0.05)
        except Exception:
            print("zero = np.array([%s, %s])" % tuple(zero))
            print("a = np.array([%s, %s])" % tuple(a))
            print("b = np.array([%s, %s])" % tuple(b))

            print("zzero = np.array([%s, %s])" % tuple(zzero))
            print("aa = np.array([%s, %s])" % tuple(aa))
            print("bb = np.array([%s, %s])" % tuple(bb))
            raise


def test_run_refine_sparse(lt_ctx):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    matcher = grm.Matcher()
    match_pattern = blobfinder.RadialGradient(radius=radius)

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    (res, real_indices) = blobfinder.run_refine(
        ctx=lt_ctx,
        dataset=dataset,
        zero=zero + np.random.uniform(-0.5, 0.5, size=2),
        a=a + np.random.uniform(-0.5, 0.5, size=2),
        b=b + np.random.uniform(-0.5, 0.5, size=2),
        matcher=matcher,
        match_pattern=match_pattern,
        correlation='sparse',
        steps=3
    )

    print(peaks - grm.calc_coords(
        res['zero'].data[0],
        res['a'].data[0],
        res['b'].data[0],
        indices)
    )

    assert np.allclose(res['zero'].data[0], zero, atol=0.5)
    assert np.allclose(res['a'].data[0], a, atol=0.2)
    assert np.allclose(res['b'].data[0], b, atol=0.2)


def test_custom_template():
    template = m.radial_gradient(centerX=10, centerY=10, imageSizeX=21, imageSizeY=23, radius=7)
    custom = blobfinder.UserTemplate(template=template, search=18)

    assert custom.get_crop_size() == 18

    same = custom.get_mask((23, 21))
    larger = custom.get_mask((25, 23))
    smaller = custom.get_mask((10, 10))

    assert np.allclose(same, template)
    assert np.allclose(larger[1:-1, 1:-1], template)
    assert np.allclose(template[6:-7, 5:-6], smaller)


def test_custom_template_fuzz():
    for i in range(10):
        integers = np.arange(1, 15)
        center_y = np.random.choice(integers)
        center_x = np.random.choice(integers)

        size_y = np.random.choice(integers)
        size_x = np.random.choice(integers)

        radius = np.random.choice(integers)
        search = np.random.choice(integers)

        mask_y = np.random.choice(integers)
        mask_x = np.random.choice(integers)

        print("center_y:", center_y)
        print("center_x:", center_x)
        print("size_y:", size_y)
        print("size_x:", size_x)
        print("radius:", radius)
        print("search:", search)
        print("mask_y:", mask_y)
        print("mask_x:", mask_x)

        template = m.radial_gradient(
            centerX=center_x, centerY=center_y,
            imageSizeX=size_x, imageSizeY=size_y,
            radius=radius
        )
        custom = blobfinder.UserTemplate(template=template, search=search)

        mask = custom.get_mask((mask_y, mask_x))  # noqa


def test_featurevector(lt_ctx):
    shape = np.array([128, 128])
    zero = shape // 2
    a = np.array([24, 0.])
    b = np.array([0., 30])
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 5
    radius_outer = 10

    template = m.background_subtraction(
        centerX=radius_outer + 1,
        centerY=radius_outer + 1,
        imageSizeX=radius_outer*2 + 2,
        imageSizeY=radius_outer*2 + 2,
        radius=radius_outer,
        radius_inner=radius + 1,
        antialiased=False
    )

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius, all_equal=True)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    match_pattern = blobfinder.UserTemplate(template=template)

    stack = functools.partial(
        blobfinder.feature_vector,
        imageSizeX=shape[1],
        imageSizeY=shape[0],
        peaks=peaks,
        match_pattern=match_pattern,

    )

    job = lt_ctx.create_mask_job(
        dataset=dataset, factories=stack, mask_count=len(peaks), mask_dtype=np.float32
    )
    res = lt_ctx.run(job)

    peak_data, _, _ = cbed_frame(*shape, zero, a, b, np.array([(0, 0)]), radius, all_equal=True)
    peak_sum = peak_data.sum()

    assert np.allclose(res.sum(), data.sum())
    assert np.allclose(res, peak_sum)


@pytest.mark.parametrize(
    "cls,dtype,kwargs",
    [
        (blobfinder.FastCorrelationUDF, np.int, {}),
        (blobfinder.FastCorrelationUDF, np.float, {}),
        (blobfinder.SparseCorrelationUDF, np.int, {'steps': 3}),
        (blobfinder.SparseCorrelationUDF, np.float, {'steps': 3}),
    ]
)
def test_peak_input_types(lt_ctx, cls, dtype, kwargs):
    shape = np.array([128, 128])
    zero = shape / 2 + np.random.uniform(-1, 1, size=2)
    a = np.array([27.17, 0.]) + np.random.uniform(-1, 1, size=2)
    b = np.array([0., 29.19]) + np.random.uniform(-1, 1, size=2)
    indices = np.mgrid[-2:3, -2:3]
    indices = np.concatenate(indices.T)

    radius = 10

    data, indices, peaks = cbed_frame(*shape, zero, a, b, indices, radius)

    dataset = MemoryDataSet(data=data, tileshape=(1, *shape),
                            num_partitions=1, sig_dims=2)

    template = m.radial_gradient(
        centerX=radius+1,
        centerY=radius+1,
        imageSizeX=2*radius+2,
        imageSizeY=2*radius+2,
        radius=radius
    )

    match_patterns = [
        blobfinder.RadialGradient(radius=radius),
        blobfinder.Circular(radius=radius),
        blobfinder.BackgroundSubtraction(radius=radius),
        blobfinder.RadialGradientBackgroundSubtraction(radius=radius),
        blobfinder.UserTemplate(template=template)
    ]

    print("zero: ", zero)
    print("a: ", a)
    print("b: ", b)

    for match_pattern in match_patterns:
        print("refining using template %s" % type(match_pattern))
        udf = cls(match_pattern=match_pattern, peaks=peaks.astype(dtype), **kwargs)
        res = lt_ctx.run_udf(dataset=dataset, udf=udf)
        print(peaks - res['refineds'].data[0])

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # plt.imshow(data[0])
        # for p in np.flip(res['refineds'].data[0], axis=-1):
        #     ax.add_artist(plt.Circle(p, radius, fill=False, color='y'))
        # plt.show()

        assert np.allclose(res['refineds'].data[0], peaks, atol=0.5)
