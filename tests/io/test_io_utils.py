from libertem.io.utils import get_partition_shape
from libertem.common import Shape


def test_partition_shape_1d():
    ds_shape = Shape((15, 16, 16), sig_dims=2)
    pshape = get_partition_shape(
        dataset_shape=ds_shape,
        target_size_items=256*1024,
        min_num=2
    )
    assert pshape == (7,)


def test_partition_shape_1():
    assert get_partition_shape(Shape((15, 16, 16), sig_dims=2), target_size_items=512) == (
        (2,)
    )


def test_partition_shape_2():
    assert get_partition_shape(Shape((1, 15, 16, 16), sig_dims=2), target_size_items=512) == (
        (1, 2,)
    )


def test_partition_shape_3():
    assert get_partition_shape(Shape((15, 15, 16, 16), sig_dims=2), target_size_items=15*512) == (
        (2, 15,)
    )


def test_partition_shape_4():
    assert get_partition_shape(
        Shape((128, 15, 15, 16, 16), sig_dims=2),
        target_size_items=15*512
    ) == (
        (1, 2, 15,)
    )


def test_partition_shape_5():
    assert get_partition_shape(
        Shape((2, 16, 16), sig_dims=2),
        target_size_items=512,
        min_num=3
    ) == (
        (1,)
    )


def test_partition_shape_small():
    assert get_partition_shape(Shape((15, 16, 16), sig_dims=2), target_size_items=4) == (
        (1,)
    )
