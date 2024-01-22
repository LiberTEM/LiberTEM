from libertem.web.helpers import _convert_device_map


def test_convert_device_map():
    assert _convert_device_map({0: 3, 1: 0}) == [0, 0, 0]
    assert _convert_device_map({0: 1, 1: 1}) == [0, 1]
    assert _convert_device_map({0: 0, 1: 1, 2: 2}) == [1, 2, 2]
