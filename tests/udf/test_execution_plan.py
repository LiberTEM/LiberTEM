from collections import OrderedDict
import pytest

from sparseconverter import (
    BACKENDS, CPU_BACKENDS, CUPY, CUPY_SCIPY_COO, NUMPY, SCIPY_COO, SCIPY_CSC, SCIPY_CSR,
    SPARSE_COO, SPARSE_GCXS, get_backend
)
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.base import _execution_plan
from libertem.udf.base import UDF, _get_canonical_backends
from libertem.utils.devices import detect

from utils import set_device_class


d = detect()
has_cupy = d['cudas'] and d['has_cupy']
has_gpus = len(d['cudas']) > 0


class BackendUDF(UDF):
    def __init__(
        self, preferred_input_dtype=UDF.USE_NATIVE_DTYPE,
        array_backends=UDF.BACKEND_ALL
    ) -> None:
        super().__init__(preferred_input_dtype=preferred_input_dtype, array_backends=array_backends)

    def process_tile(self, tile):
        assert get_backend(tile) in self.get_backends()
        assert get_backend(tile) == self.meta.array_backend
        self.results.backends[0].append((get_backend(tile), self.meta.slice))

    def preprocess(self):
        self.results.backends[0] = []

    def merge(self, dest, src):
        dest.backends[0].append(src.backends[0])

    def get_result_buffers(self):
        return {
            'backends': self.buffer(kind='single', dtype=object, extra_shape=(1,))
        }

    def get_preferred_input_dtype(self):
        '''
        Return the value passed in the constructor.
        '''
        return self.params.preferred_input_dtype

    def get_backends(self):
        return self.params.array_backends


def canonical_plan(execution_plan):
    res = OrderedDict()
    for key, value in execution_plan.items():
        res[key] = frozenset(value)
    return res


def validate(ctx, ds, udfs, available_backends, source_backend, execution_plan, do_run=True):
    all_udfs = set()
    for b, selected_udfs in execution_plan.items():
        for udf in selected_udfs:
            assert b in _get_canonical_backends(udf.get_backends())
            assert b in available_backends
            assert udf not in all_udfs
            all_udfs.add(udf)
    assert set(udfs) == all_udfs
    assert source_backend in ds.array_backends
    if do_run:
        execute(ctx, ds, udfs, available_backends, source_backend, execution_plan)


def execute(ctx, ds, udfs, available_backends, source_backend, execution_plan):
    if source_backend in CPU_BACKENDS and all(key in CPU_BACKENDS for key in execution_plan):
        device_class = 'cpu'
    else:
        device_class = 'cupy'
    class_for_udf = {}
    for key, udfs in execution_plan.items():
        for udf in udfs:
            class_for_udf[udf] = key

    if available_backends != BACKENDS:
        run_backends = tuple(available_backends)
    else:
        run_backends = None

    with set_device_class(device_class):
        res = ctx.run_udf(dataset=ds, udf=udfs, backends=run_backends)
        for r, udf in zip(res, udfs):
            backends = r['backends'].data[0]
            for p in backends:
                for b in p:
                    assert b[0] == class_for_udf[udf]


@pytest.mark.parametrize(
    # Only test the "proper" backends, leave out
    # DOK, NumPy matrix and CUDA
    'left', UDF.BACKEND_ALL
)
@pytest.mark.parametrize(
    # Only test the "proper" backends, leave out
    # DOK, NumPy matrix and CUDA
    'right', UDF.BACKEND_ALL
)
@pytest.mark.parametrize(
    'device_class', (None, 'cpu', 'cuda')
)
def test_simple(lt_ctx, left, right, device_class):
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=(left, ))
    udf1 = BackendUDF(array_backends=(right,))
    source_backend, execution_plan = _execution_plan(
        udfs=[udf1],
        ds=ds,
        device_class=device_class,
        available_backends=BACKENDS,
    )
    assert source_backend == left
    assert execution_plan == {right: [udf1]}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf1],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


@pytest.mark.parametrize(
    'right', UDF.BACKEND_ALL
)
def test_direct_path_ds(lt_ctx, right):

    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=UDF.BACKEND_ALL)
    udf1 = BackendUDF(array_backends=(right,))
    udf2 = BackendUDF(array_backends=UDF.BACKEND_ALL)
    source_backend, execution_plan = _execution_plan(
        udfs=[udf2, udf1],
        ds=ds,
        device_class=None,
        available_backends=BACKENDS,
    )
    assert source_backend == right
    assert canonical_plan(execution_plan) == {right: frozenset([udf1, udf2])}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf1, udf2],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


@pytest.mark.parametrize(
    'left', UDF.BACKEND_ALL
)
def test_direct_path_udf(lt_ctx, left):
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=(left, ))
    udf = BackendUDF(array_backends=UDF.BACKEND_ALL)
    source_backend, execution_plan = _execution_plan(
        udfs=[udf],
        ds=ds,
        device_class=None,
        available_backends=BACKENDS,
    )
    assert source_backend == left
    assert execution_plan == {left: [udf]}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


@pytest.mark.parametrize(
    'available', UDF.BACKEND_ALL
)
@pytest.mark.parametrize(
    'device_class', (None, 'cpu', 'cuda')
)
def test_available(lt_ctx, available, device_class):
    # Verify that the choice of backends can be restricted to available options
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=UDF.BACKEND_ALL)
    udf = BackendUDF(array_backends=UDF.BACKEND_ALL)
    source_backend, execution_plan = _execution_plan(
        udfs=[udf],
        ds=ds,
        device_class=device_class,
        available_backends=(available, ),
    )
    assert source_backend == available
    assert execution_plan == {available: [udf]}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf],
        available_backends=(available, ),
        source_backend=source_backend,
        execution_plan=execution_plan
    )


def test_ds_preference(lt_ctx):
    # Two direct paths and not clear which is better,
    # we pick the first one of the dataset
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=(SCIPY_COO, NUMPY))
    udf = BackendUDF(array_backends=UDF.BACKEND_ALL)
    source_backend, execution_plan = _execution_plan(
        udfs=[udf],
        ds=ds,
        device_class=None,
        available_backends=BACKENDS,
    )
    assert source_backend == SCIPY_COO
    assert execution_plan == {SCIPY_COO: [udf]}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


def test_udf_preference_sparse(lt_ctx):
    # The dataset has both dense and sparse for CPU
    # We include a format with fast conversion from sparse
    # CPU to sparse GPU so that it wins over NumPy
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=(SPARSE_COO, SCIPY_COO, NUMPY))
    # The UDF has both dense and sparse on GPU and CPU
    udf = BackendUDF(array_backends=(NUMPY, SCIPY_COO, CUPY, CUPY_SCIPY_COO))
    source_backend, execution_plan = _execution_plan(
        udfs=[udf],
        ds=ds,
        # We assign preferentially for GPU to trigger
        # conversion and not pick a direct option
        # unde rthe assumption that the transfer and conversion
        # cost is offset by processing speed
        # If the sparse->sparse converter is inefficient this will
        # actually do dense-> sparse!
        device_class='cuda',
        available_backends=BACKENDS,
    )
    # We use the sparse option
    # SPARSE_COO -> CUPY_SCIPY_COO is currently
    # expensive, so NUMPY would be chosen here
    assert source_backend == SCIPY_COO
    assert execution_plan == {CUPY_SCIPY_COO: [udf]}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


def test_udf_preference_dense(lt_ctx):
    # The dataset has only dense for CPU.
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=(NUMPY, ))
    # The UDF has both dense and sparse on GPU
    # Indicating preference for sparse
    udf = BackendUDF(array_backends=(NUMPY, CUPY_SCIPY_COO, CUPY))
    source_backend, execution_plan = _execution_plan(
        udfs=[udf],
        ds=ds,
        # We assign preferentially for GPU to trigger
        # transfer and not pick the direct NUMPY option
        device_class='cuda',
        available_backends=BACKENDS,
    )
    # We use the dense option to avoid dense->sparse conversion
    assert source_backend == NUMPY
    assert execution_plan == {CUPY: [udf]}
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


def test_udf_preference_number_order(lt_ctx):
    # The dataset has only dense for CPU.
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=(NUMPY, ))
    # The UDF has only sparse on CPU
    # Choose by order or occurrence
    udf1 = BackendUDF(array_backends=(SCIPY_CSR, SCIPY_CSC, SCIPY_COO))
    udf2 = BackendUDF(array_backends=(SCIPY_CSR, SCIPY_COO))
    udf3 = BackendUDF(array_backends=(SPARSE_GCXS, SPARSE_COO))
    source_backend, execution_plan = _execution_plan(
        udfs=[udf1, udf2, udf3],
        ds=ds,
        device_class=None,
        available_backends=BACKENDS,
    )
    # We use SCIPY_CSR (fast conversion, more UDFs, before SCIPY_COO),
    # then SPARSE_GCXS (slower conversion), and not SPARSE_COO (less preferred)
    assert source_backend == NUMPY
    assert canonical_plan(execution_plan) == {
        SCIPY_CSR: frozenset([udf1, udf2]),
        SPARSE_GCXS: frozenset([udf3]),
    }
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf1, udf2, udf3],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan
    )


@pytest.mark.parametrize(
    'device_class', (None, 'cpu', 'cuda')
)
@pytest.mark.parametrize(
    'ds_backends', [(CUPY, NUMPY), (NUMPY, CUPY), (NUMPY, ), (CUPY, )]
)
@pytest.mark.parametrize(
    'udf4_backends', [(SCIPY_CSC, SPARSE_GCXS), (SPARSE_GCXS, SCIPY_CSC)]
)
def test_udf_preference_number(lt_ctx, device_class, ds_backends, udf4_backends):
    # The dataset has only dense for CPU or GPU.
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=ds_backends)
    # The UDFs have only sparse on CPU
    # Choose by number of UDFs and order of occurrence
    udf1 = BackendUDF(array_backends=(SCIPY_CSR, SCIPY_CSC, SCIPY_COO))
    udf2 = BackendUDF(array_backends=(SCIPY_CSR, SPARSE_COO, SCIPY_COO))
    udf3 = BackendUDF(array_backends=(SPARSE_GCXS, SCIPY_COO))
    udf4 = BackendUDF(array_backends=udf4_backends)
    source_backend, execution_plan = _execution_plan(
        udfs=[udf1, udf2, udf3, udf4],
        ds=ds,
        device_class=device_class,
        available_backends=BACKENDS,
    )
    print(source_backend, execution_plan)
    # Prefer options without CPU-GPU conversion
    assert source_backend == NUMPY if NUMPY in ds_backends else CUPY
    # We use SCIPY_COO (more UDFs) despite being less preferred
    # by the UDFs.
    # SCIPY_COO comes first since the converter is faster
    # than SPARSE_GCXS
    # FIXME There might
    # be faster options that require only two conversions as well
    assert canonical_plan(execution_plan) == {
        SCIPY_COO: frozenset([udf1, udf2, udf3]),
        udf4_backends[0]: frozenset([udf4]),
    }
    needs_cupy = device_class in ('cupy', 'cuda') or CUPY in ds_backends
    do_run = not needs_cupy or has_cupy
    validate(
        ctx=lt_ctx,
        ds=ds,
        udfs=[udf1, udf2, udf3, udf4],
        available_backends=BACKENDS,
        source_backend=source_backend,
        execution_plan=execution_plan,
        do_run=do_run,
    )


def test_no_solution_udf():
    ds = MemoryDataSet(datashape=(2, 2, 2, 2), array_backends=UDF.BACKEND_ALL)
    # only GPU for UDF
    udf = BackendUDF(array_backends=(CUPY, ))
    with pytest.raises(RuntimeError):
        source_backend, execution_plan = _execution_plan(
            udfs=[udf],
            ds=ds,
            device_class=None,
            # restrict to CPU only
            available_backends=CPU_BACKENDS,
        )
