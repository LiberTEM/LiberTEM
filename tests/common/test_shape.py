from libertem.common import Shape


def test_shape_get_nav():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert tuple(s.nav) == (16, 16)


def test_shape_get_sig():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert tuple(s.sig) == (128, 128)


def test_shape_get_size():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert s.size == 16 * 16 * 128 * 128


def test_shape_flatten_nav():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert tuple(s.flatten_nav()) == (16 * 16, 128, 128)


def test_shape_flatten_sig():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert tuple(s.flatten_sig()) == (16, 16, 128 * 128)


def test_shape_getitem():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert s[:2] == (16, 16)
    assert s[0] == 16


def test_shape_len():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert len(s) == 4


def test_shape_to_tuple():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert s.to_tuple() == (16, 16, 128, 128)


def test_shape_repr():
    s = Shape((16, 16, 128, 128), sig_dims=2)
    assert repr(s) == "(16, 16, 128, 128)"


def test_shape_eq_1():
    s1 = Shape((16, 16, 128, 128), sig_dims=2)
    s2 = Shape((16, 16, 128, 128), sig_dims=2)
    assert s1 == s2


def test_shape_eq_2():
    s1 = Shape((16, 16, 128, 128), sig_dims=2)
    s2 = Shape((16, 16, 128, 128), sig_dims=3)
    assert s1 != s2


def test_shape_eq_3():
    s1 = Shape((16, 16, 128, 128), sig_dims=2)
    s2 = Shape((16 * 16, 128, 128), sig_dims=2)
    assert s1 != s2


def test_shape_eq_4():
    s1 = Shape((17, 16, 128, 128), sig_dims=2)
    s2 = Shape((16, 16, 128, 128), sig_dims=2)
    assert s1 != s2
