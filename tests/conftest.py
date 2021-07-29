import os
import shutil

import pytest

import utils


def _raw_on_workers(tmpdirpath, datadir, filename, ctx):
    data = utils._mk_random(size=(16, 16, 128, 128), dtype='float32')

    def _make_example_raw():
        # workers don't automatically have the pytest tmp directory, create it:
        if not os.path.exists(tmpdirpath):
            os.makedirs(tmpdirpath)
        print("creating %s" % filename)
        data.tofile(filename)
        print("created %s" % filename)
        return tmpdirpath, os.listdir(tmpdirpath)

    import cloudpickle
    import pickle
    dumped = cloudpickle.dumps(_make_example_raw)

    pickle.loads(dumped)

    print("raw_on_workers _make_example_raw: %s" %
          (ctx.executor.run_each_host(_make_example_raw),))

    ds = ctx.load("raw",
                  path=str(filename),
                  nav_shape=(16, 16),
                  sig_shape=(128, 128),
                  dtype="float32")
    return ds


@pytest.fixture
def raw_on_workers(dist_ctx, tmpdir_factory):
    """
    copy raw dataset to each worker
    """

    datadir = tmpdir_factory.mktemp('data')
    filename = str(datadir + '/raw-test-on-workers')
    tmpdirpath = os.path.dirname(filename)

    ds = _raw_on_workers(tmpdirpath, datadir, filename, dist_ctx)
    yield ds

    def _cleanup():
        # NOTE: can't call a function defined in conftest here, as `conftest` is not available
        # as a module on the worker nodes
        files = os.listdir(tmpdirpath)
        shutil.rmtree(tmpdirpath, ignore_errors=True)
        print("removed %s" % tmpdirpath)
        return tmpdirpath, files

    print(f"raw_on_workers cleanup: {dist_ctx.executor.run_each_host(_cleanup)}")


@pytest.fixture
def raw_on_workers_ipy(ipy_ctx, tmpdir_factory):
    """
    copy raw dataset to each worker
    """

    datadir = tmpdir_factory.mktemp('data')
    filename = str(datadir + '/raw-test-on-workers')
    tmpdirpath = os.path.dirname(filename)

    ds = _raw_on_workers(tmpdirpath, datadir, filename, ipy_ctx)
    yield ds

    def _cleanup():
        # NOTE: can't call a function defined in conftest here, as `conftest` is not available
        # as a module on the worker nodes
        files = os.listdir(tmpdirpath)
        shutil.rmtree(tmpdirpath, ignore_errors=True)
        print("removed %s" % tmpdirpath)
        return tmpdirpath, files

    print(f"raw_on_workers cleanup: {ipy_ctx.executor.run_each_host(_cleanup)}")
