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
                  scan_size=(16, 16),
                  detector_size=(128, 128),
                  dtype="float32")
    return ds


def _raw_on_workers_cleanup(tmpdirpath):
    # FIXME: this may litter /tmp/ with empty directories, as we only remove our own
    # tmpdirpath, but as we run these tests in docker containers, they are eventually
    # cleaned up anyways:
    files = os.listdir(tmpdirpath)
    shutil.rmtree(tmpdirpath, ignore_errors=True)
    print("removed %s" % tmpdirpath)
    return tmpdirpath, files


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
        _raw_on_workers_cleanup(tmpdirpath)

    print("raw_on_workers cleanup: %s" % (dist_ctx.executor.run_each_host(_cleanup),))


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
        _raw_on_workers_cleanup(tmpdirpath)

    print("raw_on_workers cleanup: %s" % (ipy_ctx.executor.run_each_host(_cleanup),))
