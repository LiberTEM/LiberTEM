import numpy as np
import sparse
import pytest

from libertem.io.corrections import CorrectionSet, detector
from libertem.utils.generate import gradient_data, exclude_pixels
from libertem.udf.base import NoOpUDF
from libertem.api import Context


# Adust to scale benchmark:
COMMON_ROI = np.s_[:10, :10]


@pytest.fixture(scope='module')
def mod_ctx():
    """
    To make it easy to experiment with different executors
    and their parameters, we have a local fixture here.
    """
    from libertem.executor.pipelined import PipelinedExecutor
    from libertem.utils.devices import detect
    specargs = detect()
    specargs.update({'cudas': []})
    spec = PipelinedExecutor.make_spec(**specargs)
    executor = PipelinedExecutor(spec=spec, pin_workers=True)
    yield Context(executor=executor)
    # yield Context.make_with(gpus=0)
    # yield Context.make_with('pipelined', gpus=0)


@pytest.mark.benchmark(
    group="adjust tileshape",
)
@pytest.mark.parametrize(
    "base_shape", ((1, 1), (2, 2))
)
@pytest.mark.parametrize(
    "excluded_coords", (
        # These magic numbers are "worst case" to produce collisions
        # 2*3*4*5*6*7
        np.array([
            (720, 210, 306),
            (120, 210, 210)
        ]),
        # Diagonal that prevents any tiling
        np.array([
            range(1024),
            range(1024),
        ]),
        # Column that prevents tiling in one dimension
        np.array([
            range(1024),
            np.full(1024, 3),
        ])
    )
)
def test_tileshape_adjustment_bench(benchmark, base_shape, excluded_coords):
    sig_shape = (1024, 1024)
    tile_shape = base_shape
    excluded_pixels = sparse.COO(coords=excluded_coords, shape=sig_shape, data=True)
    corr = CorrectionSet(excluded_pixels=excluded_pixels)
    adjusted = benchmark(
        corr.adjust_tileshape,
        tile_shape=tile_shape, sig_shape=sig_shape, base_shape=base_shape
    )
    print("Base shape", base_shape)
    print("Excluded coords", excluded_coords)
    print("Adjusted", adjusted)


@pytest.mark.benchmark(
    group="patch many",
)
@pytest.mark.parametrize(
    "num_excluded", (0, 1, 10, 100, 1000, 10000)
)
def test_detector_patch_large(num_excluded, benchmark):
    nav_dims = (8, 8)
    sig_dims = (1336, 2004)

    data = gradient_data(nav_dims, sig_dims)

    exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=num_excluded)

    damaged_data = data.copy()

    if exclude is not None:
        assert exclude.shape[1] == num_excluded
        damaged_data[(Ellipsis, *exclude)] = 1e24

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)
    print("Exclude: ", exclude)

    benchmark(
        detector.correct,
        buffer=damaged_data,
        excluded_pixels=exclude,
        sig_shape=sig_dims,
        inplace=False
    )


@pytest.mark.benchmark(
    group="correct large",
)
def test_detector_correction_large(benchmark):
    nav_dims = (8, 8)
    sig_dims = (1336, 2004)

    data = gradient_data(nav_dims, sig_dims)
    gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
    dark_image = np.random.random(sig_dims).astype(np.float64)

    damaged_data = data.copy()
    damaged_data /= gain_map
    damaged_data += dark_image

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)

    benchmark(
        detector.correct,
        buffer=damaged_data,
        dark_image=dark_image,
        gain_map=gain_map,
        sig_shape=sig_dims,
        inplace=False,
    )


@pytest.mark.benchmark(
    group="corrset creation",
)
@pytest.mark.parametrize(
    'num_excluded', (0, 1, 10, 100, 1000, 10000, 100000)
)
def test_descriptor_creation(num_excluded, benchmark):
    shape = (2000, 2000)
    excluded_coords = (
        np.random.randint(0, 2000, num_excluded),
        np.random.randint(0, 2000, num_excluded),
    )
    benchmark(
        detector.RepairDescriptor,
        sig_shape=shape,
        excluded_pixels=excluded_coords,
        allow_empty=True
    )


class TestRealCorrection:
    @pytest.mark.benchmark(
        group="correct large",
    )
    def test_real_correction_baseline(self, mod_ctx, large_raw_file, benchmark):
        filename, shape, dtype = large_raw_file
        nav_dims = shape[:2]
        sig_dims = shape[2:]

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)

        udf = NoOpUDF()
        ds = mod_ctx.load(
            'RAW',
            path=str(filename),
            nav_shape=shape[:2],
            dtype=dtype,
            sig_shape=shape[2:],
        )

        benchmark(
            mod_ctx.run_udf,
            dataset=ds,
            udf=udf,
            roi=ds.roi[COMMON_ROI],
        )

    @pytest.mark.benchmark(
        group="correct large",
    )
    @pytest.mark.parametrize(
        'gain', ('no gain', 'use gain')
    )
    @pytest.mark.parametrize(
        'dark', ('no dark', 'use dark')
    )
    @pytest.mark.parametrize(
        'num_excluded', (0, 1, 1000, 10000)
    )
    def test_real_correction(self, mod_ctx, large_raw_file, benchmark,
            gain, dark, num_excluded):
        filename, shape, dtype = large_raw_file
        nav_dims = shape[:2]
        sig_dims = shape[2:]

        if gain == 'use gain':
            gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        elif gain == 'no gain':
            gain_map = None
        else:
            raise ValueError

        if dark == 'use dark':
            dark_image = np.random.random(sig_dims).astype(np.float64)
        elif dark == 'no dark':
            dark_image = None
        else:
            raise ValueError

        if num_excluded > 0:
            excluded_coords = exclude_pixels(sig_dims=sig_dims, num_excluded=num_excluded)
            assert excluded_coords is not None
            assert excluded_coords.shape[1] == num_excluded
            exclude = sparse.COO(coords=excluded_coords, shape=sig_dims, data=True)
        else:
            exclude = None

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)

        corrset = CorrectionSet(
            dark=dark_image,
            gain=gain_map,
            excluded_pixels=exclude,
        )

        udf = NoOpUDF()
        ds = mod_ctx.load(
            'RAW',
            path=str(filename),
            nav_shape=shape[:2],
            dtype=dtype,
            sig_shape=shape[2:],
        )

        benchmark(
            mod_ctx.run_udf,
            dataset=ds,
            udf=udf,
            corrections=corrset,
            roi=ds.roi[COMMON_ROI],
        )
