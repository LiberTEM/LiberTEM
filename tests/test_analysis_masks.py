import pytest
import numpy as np
import scipy.sparse as sp
import sparse

from utils import _naive_mask_apply, _mk_random

from libertem.common.sparse import to_dense, to_sparse, is_sparse
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.common import Shape, Slice
from libertem.utils.devices import detect
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf import UDF, UDFMeta


def _run_mask_test_program(lt_ctx, dataset, mask, expected):
    dtype = UDF.USE_NATIVE_DTYPE

    analysis_default = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask], dtype=dtype
    )
    analysis_sparse = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: to_sparse(mask)], use_sparse=True,
        dtype=dtype
    )
    analysis_dense = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: to_dense(mask)], use_sparse=False,
        dtype=dtype
    )
    results_default = lt_ctx.run(analysis_default)
    results_sparse = lt_ctx.run(analysis_sparse)
    results_dense = lt_ctx.run(analysis_dense)

    assert np.allclose(
        results_default.mask_0.raw_data,
        expected
    )
    assert np.allclose(
        results_sparse.mask_0.raw_data,
        expected
    )
    assert np.allclose(
        results_dense.mask_0.raw_data,
        expected
    )


@pytest.mark.xfail
@pytest.mark.slow
def test_weird_partition_shapes_1_slow(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), partition_shape=(16, 16, 2, 2))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)

    p = next(dataset.get_partitions())
    t = next(p.get_tiles())
    assert tuple(t.tile_slice.shape) == (1, 1, 2, 2)


@pytest.mark.xfail
def test_weird_partition_shapes_1_fast(lt_ctx):
    # XXX MemoryDataSet is now using Partition3D and so on, so we can't create
    # partitions with weird shapes so easily anymore (in this case, partitioned in
    # the signal dimensions). maybe fix this with a custom DataSet impl that simulates this?
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), partition_shape=(16, 16, 8, 8))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)

    p = next(dataset.get_partitions())
    t = next(p.get_tiles())
    assert tuple(t.tile_slice.shape) == (1, 8, 8, 8)


def test_normal_partition_shape(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_single_frame_tiles(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.slow
def test_subframe_tiles_slow(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 4, 4), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_subframe_tiles_fast(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(8, 4, 4), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_mask_uint(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16)).astype("uint16")
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_endian(lt_ctx):
    data = np.random.choice(a=0xFFFF, size=(16, 16, 16, 16)).astype(">u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_signed(lt_ctx):
    data = np.random.choice(a=0xFFFF, size=(16, 16, 16, 16)).astype("<i4")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    # NOTE: we allow casting from int32 to float32 here, and may lose some
    # precision in case of data with large dynamic range
    dataset = MemoryDataSet(
        data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
        check_cast=False,
    )

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_masks(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        mask0 = _mk_random(size=(16, 16))
        mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))
        mask2 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
        expected = _naive_mask_apply([mask0, mask1, mask2], data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=[lambda: mask0, lambda: mask1, lambda: mask2],
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
        assert np.allclose(
            results.mask_2.raw_data,
            expected[2],
        )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_mask_stack_dense(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, mask_count=2,
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_mask_stack_sparse(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = sparse.COO.from_numpy(_mk_random(size=(2, 16, 16)))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, mask_count=2,
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_mask_stack_force_sparse(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse=True, mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
@pytest.mark.with_numba  # coverage for rmatmul implementation
def test_multi_mask_stack_force_scipy_sparse(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='scipy.sparse', mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
@pytest.mark.with_numba  # coverage for rmatmul implementation
def test_multi_mask_stack_force_scipy_sparse_csc(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='scipy.sparse.csc', mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_mask_stack_force_sparse_pydata(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='sparse.pydata', mask_count=2
        )
        if backend == 'cupy':
            with pytest.raises(ValueError):
                results = lt_ctx.run(analysis)
        else:
            results = lt_ctx.run(analysis)
            assert np.allclose(
                results.mask_0.raw_data,
                expected[0],
            )
            assert np.allclose(
                results.mask_1.raw_data,
                expected[1],
            )
    finally:
        set_use_cpu(0)


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_mask_stack_force_dense(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = sparse.COO.from_numpy(_mk_random(size=(2, 16, 16)))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse=False, mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
    finally:
        set_use_cpu(0)


def test_multi_mask_autodtype(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    masks = _mk_random(size=(2, 16, 16))
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype == np.result_type(np.float32, data.dtype, masks.dtype)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_multi_mask_autodtype_wide(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="int64")
    masks = _mk_random(size=(2, 16, 16))
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype == np.result_type(np.float64, data.dtype, masks.dtype)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_multi_mask_autodtype_complex(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="complex64")
    masks = _mk_random(size=(2, 16, 16))
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(dataset=dataset, factories=lambda: masks)
    results = lt_ctx.run(analysis)

    assert results.mask_0_complex.raw_data.dtype.kind == 'c'
    assert results.mask_0_complex.raw_data.dtype == np.complex64

    assert np.allclose(
        results.mask_0_complex.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1_complex.raw_data,
        expected[1],
    )


def test_multi_mask_autodtype_complex_wide(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    masks = _mk_random(size=(2, 16, 16), dtype="complex128")
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0_complex.raw_data.dtype.kind == 'c'
    assert results.mask_0_complex.raw_data.dtype == np.complex128

    assert np.allclose(
        results.mask_0_complex.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1_complex.raw_data,
        expected[1],
    )


def test_multi_mask_force_dtype(lt_ctx):
    force_dtype = np.dtype(np.int32)
    data = _mk_random(size=(16, 16, 16, 16), dtype="int16")
    masks = _mk_random(size=(2, 16, 16), dtype="bool")
    expected = _naive_mask_apply(masks.astype(force_dtype), data.astype(force_dtype))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks, dtype=force_dtype
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype.kind == force_dtype.kind
    assert results.mask_0.raw_data.dtype == force_dtype

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_avoid_calculating_masks_on_client(hdf5_ds_1, local_cluster_ctx):
    mask = _mk_random(size=(16, 16))
    # We have to start a local cluster so that the masks are
    # computed in a different process
    analysis = local_cluster_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask], mask_count=1, mask_dtype=np.float32
    )
    udf = analysis.get_udf()
    local_cluster_ctx.run_udf(dataset=hdf5_ds_1, udf=udf)

    assert udf.masks._computed_masks is None


def test_avoid_calculating_masks_on_client_udf(hdf5_ds_1, local_cluster_ctx):
    mask = _mk_random(size=(16, 16))
    # We have to use a real cluster instead of InlineJobExecutor so that the masks are
    # computed in a different process
    analysis = local_cluster_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask], mask_count=1, mask_dtype=np.float32
    )
    udf = analysis.get_udf()
    local_cluster_ctx.run_udf(udf=udf, dataset=hdf5_ds_1)
    assert udf._mask_container is None


def test_override_mask_dtype(lt_ctx):
    mask_dtype = np.float32
    data = _mk_random(size=(16, 16, 16, 16), dtype=mask_dtype)
    masks = _mk_random(size=(2, 16, 16), dtype=np.float64)
    expected = _naive_mask_apply(masks.astype(mask_dtype), data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks, mask_dtype=mask_dtype, mask_count=len(masks),
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype == mask_dtype

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_mask_udf(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask2 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    # The ApplyMasksUDF returns data with shape ds.shape.nav + (mask_count, ),
    # different from ApplyMasksJob
    expected = np.moveaxis(_naive_mask_apply([mask0, mask1, mask2], data), (0, 1), (2, 0))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    udf = ApplyMasksUDF(
        mask_factories=[lambda: mask0, lambda: mask1, lambda: mask2]
    )
    results = lt_ctx.run_udf(udf=udf, dataset=dataset)

    assert np.allclose(results['intensity'].data, expected)


def test_numpy_is_sparse():
    mask = _mk_random(size=(16, 16))
    assert not is_sparse(mask)


def test_scipy_is_sparse():
    mask = sp.csr_matrix(_mk_random(size=(16, 16)))
    assert is_sparse(mask)


def test_sparse_is_sparse():
    mask = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    assert is_sparse(mask)


def test_sparse_dok_is_sparse():
    mask = sparse.DOK.from_numpy(_mk_random(size=(16, 16)))
    assert is_sparse(mask)


def test_all_sparse_analysis(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    expected = _naive_mask_apply([mask0, mask1], data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1],
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def _mask_from_analysis(dataset, analysis):
    slice_ = Slice(
        origin=(0, 0, 0),
        shape=Shape((1, 16, 16), sig_dims=2),
    )
    udf = analysis.get_udf()
    meta = UDFMeta(
        partition_slice=None,
        dataset_shape=dataset.shape,
        roi=None,
        dataset_dtype=dataset.dtype,
        input_dtype=dataset.dtype,
        corrections=None,
    )
    udf.set_meta(meta)
    return udf.masks.get(slice_, transpose=True)


def test_uses_sparse_all_default(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )

    assert is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_mixed_default(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )

    assert not is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_true(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=True
    )

    assert is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_scipy_sparse(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse='scipy.sparse'
    )

    assert sp.issparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_pydata(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse='sparse.pydata'
    )

    assert isinstance(_mask_from_analysis(dataset, analysis), sparse.SparseArray)


def test_uses_scipy_sparse_false(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=False
    )

    assert not is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_sparse_false(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    mask1 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=False
    )

    assert not is_sparse(_mask_from_analysis(dataset, analysis))


def test_masks_timeseries_2d_frames(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16, 16),
        num_partitions=2
    )
    mask0 = _mk_random(size=(16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (256,)


def test_masks_spectrum_linescan(lt_ctx):
    data = _mk_random(size=(16 * 16, 16 * 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16 * 16),
        num_partitions=2,
        sig_dims=1,
    )
    mask0 = _mk_random(size=(16 * 16, ))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16 * 16,)


def test_masks_spectrum(lt_ctx):
    data = _mk_random(size=(16, 16, 16 * 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16 * 16),
        num_partitions=2,
        sig_dims=1,
    )
    mask0 = _mk_random(size=(16 * 16, ))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16, 16)


def test_masks_hyperspectral(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16, 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16, 16),
        num_partitions=2,
        sig_dims=3,
    )
    mask0 = _mk_random(size=(16, 16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16, 16)


def test_masks_complex_ds(lt_ctx, ds_complex):
    mask0 = _mk_random(size=(16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=ds_complex, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16, 16)


def test_masks_complex_mask(lt_ctx, ds_complex):
    mask0 = _mk_random(size=(16, 16), dtype="complex64")
    analysis = lt_ctx.create_mask_analysis(
        dataset=ds_complex, factories=[lambda: mask0],
    )
    expected = _naive_mask_apply([mask0], ds_complex.data)
    results = lt_ctx.run(analysis)
    assert results.mask_0_complex.raw_data.shape == (16, 16)
    assert np.allclose(
        results.mask_0_complex.raw_data,
        expected
    )

    # also execute _run_mask_test_program to check sparse implementation.
    # _run_mask_test_program checks mask_0 result, which is np.abs(mask_0_complex)
    _run_mask_test_program(lt_ctx, ds_complex, mask0, np.abs(expected))


def test_numerics_fail(lt_ctx):
    dtype = 'float32'
    # Highest expected detector resolution
    RESOLUTION = 4096
    # Highest expected detector dynamic range
    RANGE = 1e6
    # default value for all cells
    VAL = 1.1

    data = np.full((2, 1, RESOLUTION, RESOLUTION), VAL, dtype=np.float32)
    data[0, 0, 0, 0] += VAL * RANGE
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, RESOLUTION, RESOLUTION),
        num_partitions=1,
        sig_dims=2,
    )
    mask0 = np.ones((RESOLUTION, RESOLUTION), dtype=np.float64)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0], mask_count=1, mask_dtype=dtype,
    )

    results = lt_ctx.run(analysis)
    expected = np.array([[
        [VAL*RESOLUTION**2 + VAL*RANGE],
        [VAL*RESOLUTION**2]
    ]])
    naive = _naive_mask_apply([mask0], data)
    naive_32 = _naive_mask_apply([mask0.astype(dtype)], data)
    # The masks are float64, that means the calculation is performed with high resolution
    # and the naive result should be correct
    assert np.allclose(expected, naive)
    # We make sure LiberTEM calculated this with the lower-precision dtype we set
    assert np.allclose(results.mask_0.raw_data, expected[0]) == np.allclose(naive_32, expected)
    # Confirm that the numerical precision is actually insufficient.
    # If this succeeds, we have to rethink the premise of this test.
    assert not np.allclose(results.mask_0.raw_data, expected[0])


def test_numerics_succeed(lt_ctx):
    dtype = 'float64'
    # Highest expected detector resolution
    RESOLUTION = 4096
    # Highest expected detector dynamic range
    RANGE = 1e6
    # default value for all cells
    VAL = 1.1

    data = np.full((2, 1, RESOLUTION, RESOLUTION), VAL, dtype=np.float32)
    data[0, 0, 0, 0] += VAL * RANGE
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, RESOLUTION, RESOLUTION),
        num_partitions=1,
        sig_dims=2,
    )
    mask0 = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0], mask_count=1, mask_dtype=dtype,
    )

    results = lt_ctx.run(analysis)
    expected = np.array([[
        [VAL*RESOLUTION**2 + VAL*RANGE],
        [VAL*RESOLUTION**2]
    ]])
    naive = _naive_mask_apply([mask0.astype(dtype)], data.astype(dtype))

    assert np.allclose(expected, naive)
    assert np.allclose(expected[0], results.mask_0.raw_data)
