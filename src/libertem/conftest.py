import pytest

import libertem.api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.memory import MemoryDataSet


@pytest.fixture(autouse=True)
def auto_ctx(doctest_namespace):
    ctx = lt.Context(executor=InlineJobExecutor())
    doctest_namespace["ctx"] = ctx


@pytest.fixture(autouse=True)
def auto_ds(doctest_namespace):
    dataset = MemoryDataSet(datashape=[16, 16, 16, 16])
    doctest_namespace["dataset"] = dataset


@pytest.fixture(autouse=True)
def auto_libertem(doctest_namespace):
    import libertem
    import libertem.utils
    import libertem.utils.generate
    import libertem.udf.blobfinder
    import libertem.masks
    doctest_namespace["libertem"] = libertem
    doctest_namespace["libertem.utils"] = libertem.utils
    doctest_namespace["libertem.utils.generate"] = libertem.utils.generate
    doctest_namespace["libertem.udf.blobfinder"] = libertem.udf.blobfinder
    doctest_namespace["libertem.masks"] = libertem.masks


@pytest.fixture(autouse=True)
def auto_files(doctest_namespace, hdf5, default_raw):
    doctest_namespace["path_to_hdf5"] = hdf5.filename
    doctest_namespace["path_to_raw"] = default_raw._path
