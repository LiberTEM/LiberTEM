from unittest import mock
from libertem.common import Shape, Slice
from libertem.io.dataset.base import TilingScheme


def test_tiling_scheme_methods():
    dataset_shape = Shape((16, 16, 32, 32), sig_dims=2)

    # depth=3 is chosen to not evenly divide the dataset:
    tileshape = Shape((3, 4, 32), sig_dims=2)

    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
    )

    assert len(tiling_scheme) == 8

    expected_slices = [
        Slice(
            origin=(0, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(4, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(8, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(12, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(16, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(20, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(24, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
        Slice(
            origin=(28, 0),
            shape=Shape((4, 32), sig_dims=2),
        ),
    ]
    for i in range(8):
        assert expected_slices[i] == tiling_scheme[i]

    assert [s[1] for s in tiling_scheme.slices] == expected_slices
    assert tiling_scheme.shape == tileshape
    assert tiling_scheme.dataset_shape == dataset_shape
    assert tiling_scheme.depth == 3


def test_adjust_for_partition():
    dataset_shape = Shape((16, 16, 32, 32), sig_dims=2)
    tileshape = Shape((3, 4, 32), sig_dims=2)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent="partition",
    )

    # the mismatch case:
    partition_slice = Slice(
        origin=(0, 0, 0),
        shape=Shape((4, 4, 32), sig_dims=2),
    )
    partition = mock.MagicMock()
    partition.slice = partition_slice

    tiling_scheme_adj = tiling_scheme.adjust_for_partition(partition)
    assert tiling_scheme_adj != tiling_scheme
    assert tiling_scheme_adj.depth == 4

    # the match case:
    partition_slice = Slice(
        origin=(0, 0, 0),
        shape=Shape((3, 4, 32), sig_dims=2),
    )
    partition = mock.MagicMock()
    partition.slice = partition_slice

    tiling_scheme_adj = tiling_scheme.adjust_for_partition(partition)
    assert tiling_scheme_adj == tiling_scheme
    assert tiling_scheme_adj is tiling_scheme
    assert tiling_scheme_adj.depth == 3
