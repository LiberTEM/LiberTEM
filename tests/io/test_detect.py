from libertem.io.dataset import detect
from libertem.executor.inline import InlineJobExecutor


def test_detection_empty_hdf5(empty_hdf5):
    executor = InlineJobExecutor()
    fn = empty_hdf5.filename
    filetype, params = detect(fn, executor=executor)
    assert filetype == "hdf5"
    params = params["parameters"]
    assert params != {}
    assert list(params.keys()) == ["path"]


def test_detection_nonempty_hdf5(hdf5_ds_1):
    executor = InlineJobExecutor()
    fn = hdf5_ds_1.path
    filetype, params = detect(fn, executor=executor)
    parameters = params["parameters"]
    assert parameters != {}
    assert parameters["ds_path"] == "data"
    assert parameters["path"] == fn
    assert filetype == "hdf5"
    assert list(parameters.keys()) == ["path", "ds_path", "nav_shape", "sig_shape"]
