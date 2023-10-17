import pathlib
import numpy as np
import pytest

from libertem.contrib.convert_transposed import convert_dm4_transposed, _convert_transposed_ds

from utils import _mk_random


@pytest.mark.parametrize(
    'nav_shape',
    [
        (6,),
        (3, 4),
    ]
)
@pytest.mark.parametrize(
    'sig_shape',
    [
        (8,),
        (7, 5),
    ]
)
def test_functional(lt_ctx, nav_shape, sig_shape, tmpdir_factory):
    nav_dims = len(nav_shape)
    sig_dims = len(sig_shape)

    dir = pathlib.Path(tmpdir_factory.mktemp('convert_transposed'))
    data = _mk_random(size=sig_shape + nav_shape, dtype='float32')
    # Transpose to put the nav dims before the sig dims, but retaining
    # the order of the dimensions within each group
    # e.g. (0, 1, 2, 3) => (2, 3, 0, 1)
    converted_data = np.moveaxis(
        data,
        tuple(range(data.ndim)),
        tuple((*range(nav_dims, data.ndim), *range(nav_dims))),
    )
    ds = lt_ctx.load(
        'memory',
        data=data,
        # sig_dims=nav_dims because nav_dims are at the end!
        # needed so that we correctly partition the array
        sig_dims=nav_dims,
        num_partitions=2,
    )

    convert_path = dir / 'out.npy'
    _convert_transposed_ds(lt_ctx, ds, convert_path)

    # Now that we are transposed sig_dims=sig_dims
    ds_npy = lt_ctx.load('npy', path=convert_path, sig_dims=sig_dims)
    assert ds_npy.shape.to_tuple() == converted_data.shape
    assert np.allclose(converted_data, np.load(convert_path))


def test_both_args_raise(lt_ctx):
    with pytest.raises(ValueError):
        convert_dm4_transposed(
            'tata.dm4',
            'out.npy',
            ctx=lt_ctx,
            num_cpus=42,
        )


# Tests on actual DM4 datasets are in tests/io/datasets/test_dm_single.py
# to use the dm4 file fixtures defined there
