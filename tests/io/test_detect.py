from libertem.io.dataset import detect


def test_detection_empty_hdf5(empty_hdf5):
    fn = empty_hdf5.filename
    params = detect(fn)
    assert params != {}
    assert list(params.keys()) == ["path", "type"]


def test_detection_nonempty_hdf5(hdf5_ds_1):
    fn = hdf5_ds_1.path
    params = detect(fn)
    assert params != {}
    assert params["ds_path"] == "data"
    assert params["path"] == fn
    assert params["tileshape"] == (1, 8, 16, 16)
    assert params["type"] == "hdf5"
    assert list(params.keys()) == ["path", "ds_path", "tileshape", "type"]
