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


def test_size_zero():
    s = Shape((), sig_dims=0)
    assert s.size == 0


def test_size_nav_zero():
    s = Shape((128, 128), sig_dims=2)
    assert s.nav.size == 0


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
    assert s1 != s2 and s1.sig.dims == s2.sig.dims


def test_shape_eq_4():
    s1 = Shape((17, 16, 128, 128), sig_dims=2)
    s2 = Shape((16, 16, 128, 128), sig_dims=2)
    assert s1 != s2 and s1.sig.dims == s2.sig.dims


def test_shape_add_1():
    s = Shape((12, 13, 14, 15), sig_dims=2)
    s_add = (1, 2) + s
    assert tuple(s_add) == (12, 13, 1, 2, 14, 15) and s_add.sig.dims == 2


def test_shape_add_2():
    s = Shape((12, 13, 14, 15), sig_dims=2)
    s_add = s + (1, 2)
    assert tuple(s_add) == (12, 13, 14, 15, 1, 2) and s_add.sig.dims == 4


def test_can_hash():
    shape = (12, 13, 14, 15)
    sd1 = Shape(shape, sig_dims=1)
    mapping = {sd1: 55}
    sd2 = Shape(shape, sig_dims=2)
    mapping[sd2] = 66
    assert len(mapping) == 2
    mapping.pop(sd1)
    assert sd1 not in mapping.keys()
