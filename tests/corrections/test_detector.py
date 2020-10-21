import numpy as np
import sparse
import pytest

from libertem.corrections import detector
from libertem.masks import is_sparse


def _generate_exclude_pixels(sig_dims, num_excluded):
    '''
    Generate a list of excluded pixels that
    can be reconstructed faithfully from their neighbors
    in a linear gradient dataset
    '''
    if num_excluded == 0:
        return None
    # Map of pixels that can be reconstructed faithfully from neighbors in a linear gradient
    free_map = np.ones(sig_dims, dtype=np.bool)

    # Exclude all border pixels
    for dim in range(len(sig_dims)):
        selector = tuple(slice(None) if i != dim else (0, -1) for i in range(len(sig_dims)))
        free_map[selector] = False

    exclude = []

    while len(exclude) < num_excluded:
        exclude_item = tuple([np.random.randint(low=1, high=s-1) for s in sig_dims])
        print("Exclude item: ", exclude_item)
        if free_map[exclude_item]:
            exclude.append(exclude_item)
            knock_out = tuple(slice(e - 1, e + 2) for e in exclude_item)
            # Remove the neighbors of a bad pixel
            # since that can't be reconstructed faithfully from a linear gradient
            free_map[knock_out] = False

    print("Remaining free pixel map: ", free_map)

    # Transform from list of tuples with length of number of dimensions
    # to array of indices per dimension
    return np.array(exclude).T


def _check_result(data, corrected, atol=1e-8, rtol=1e-5):
    m1 = np.unravel_index(np.argmax(np.abs(data - corrected)), data.shape)
    relative_error = np.zeros_like(data, dtype=np.result_type(data, np.float32))
    np.divide(
        np.abs(data - corrected),
        np.abs(data),
        out=relative_error,
        where=(data != 0)
    )
    m2 = np.unravel_index(
        np.argmax(relative_error), data.shape
    )
    # From np.allclose documentation
    select = np.abs(data - corrected) > (atol + rtol * np.abs(data))
    print("Maximum absolute error: data %s, corrected %s" % (data[m1], corrected[m1]))
    print("Maximum relative error: data %s, corrected %s, relative %s" %
        (data[m2], corrected[m2], relative_error[m2])
    )
    print("Triggering tolerance limits: data %s, corrected %s, relative %s" %
        (data[select],
        corrected[select],
        relative_error[select])
    )

    assert np.allclose(data, corrected, atol=atol, rtol=rtol)


def _make_data(nav_dims, sig_dims):
    data = np.linspace(
        start=5, stop=30, num=np.prod(nav_dims) * np.prod(sig_dims), dtype=np.float32
    )
    return data.reshape(nav_dims + sig_dims)


@pytest.mark.with_numba
def test_detector_correction():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([1, 2, 3])

        nav_dims = tuple(np.random.randint(low=1, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=1, high=16, size=num_sig_dims))

        data = _make_data(nav_dims, sig_dims)

        # Test pure gain and offset correction without
        # patching pixels
        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=0)

        gain_map = np.random.random(sig_dims) + 1
        dark_image = np.random.random(sig_dims)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image

        assert np.allclose((damaged_data - dark_image) * gain_map, data)

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        corrected = detector.correct(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            excluded_pixels=exclude,
            inplace=False
        )

        _check_result(
            data=data, corrected=corrected,
            atol=1e-8, rtol=1e-5
        )
        # Make sure we didn't do it in place
        assert not np.allclose(corrected, damaged_data)

        detector.correct(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            excluded_pixels=exclude,
            inplace=True
        )

        # Now damaged_data should be modified and equal to corrected
        # since it should have been done in place
        assert np.allclose(corrected, damaged_data)


@pytest.mark.with_numba
def test_detector_uint8():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([1, 2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        data = np.ones(nav_dims + sig_dims, dtype=np.uint8)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=2)

        gain_map = np.ones(sig_dims)
        dark_image = (np.random.random(sig_dims) * 3).astype(np.uint8)

        damaged_data = data.copy()
        # We don't do that since it is set to 1 above
        # damaged_data /= gain_map
        damaged_data += dark_image

        damaged_data = damaged_data.astype(np.uint8)

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        corrected = detector.correct(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            excluded_pixels=exclude,
            inplace=False
        )

        assert corrected.dtype.kind == 'f'

        _check_result(
            data=data, corrected=corrected,
            atol=1e-8, rtol=1e-5
        )
        # Make sure we didn't do it in place
        assert not np.allclose(corrected, damaged_data)

        with pytest.raises(TypeError):
            detector.correct(
                buffer=damaged_data,
                dark_image=dark_image,
                gain_map=gain_map,
                excluded_pixels=exclude,
                inplace=True
            )


@pytest.mark.with_numba
def test_detector_patch():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        data = _make_data(nav_dims, sig_dims)

        gain_map = np.random.random(sig_dims) + 1
        dark_image = np.random.random(sig_dims)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=3)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        corrected = detector.correct(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            excluded_pixels=exclude,
            inplace=False
        )

        _check_result(
            data=data, corrected=corrected,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_detector_patch_large():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = 2

        nav_dims = tuple(np.random.randint(low=3, high=5, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=4*32, high=1024, size=num_sig_dims))

        data = _make_data(nav_dims, sig_dims)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=999)

        damaged_data = data.copy()
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        corrected = detector.correct(
            buffer=damaged_data,
            excluded_pixels=exclude,
            sig_shape=sig_dims,
            inplace=False
        )

        _check_result(
            data=data, corrected=corrected,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_detector_patch_too_large():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = 2

        nav_dims = tuple(np.random.randint(low=3, high=5, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=4*32, high=1024, size=num_sig_dims))

        data = _make_data(nav_dims, sig_dims)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=1001)

        damaged_data = data.copy()
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        corrected = detector.correct(
            buffer=damaged_data,
            excluded_pixels=exclude,
            sig_shape=sig_dims,
            inplace=False
        )

        _check_result(
            data=data, corrected=corrected,
            atol=1e-8, rtol=1e-5
        )



@pytest.mark.with_numba
def test_detector_patch_overlapping():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # Faithfully reconstruct in a constant dataset
        data = np.ones(nav_dims + sig_dims)

        gain_map = np.random.random(sig_dims) + 1
        dark_image = np.random.random(sig_dims)

        # Neighboring excluded pixels
        exclude = np.ones((num_sig_dims, 3), dtype=np.int32)
        exclude[0, 1] += 1
        exclude[1, 2] += 1

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        corrected = detector.correct(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            excluded_pixels=exclude,
            inplace=False
        )

        _check_result(
            data=data, corrected=corrected,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_mask_correction():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        data = _make_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=0)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image

        assert np.allclose((damaged_data - dark_image) * gain_map, data)

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        masks = (np.random.random((2, ) + sig_dims) - 0.5).astype(np.float64)
        data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
        damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

        correct_dot = np.dot(data_flat, masks.reshape((-1, np.prod(sig_dims))).T)
        corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)

        assert not is_sparse(corrected_masks)

        reconstructed_dot =\
            np.dot(damaged_flat, corrected_masks.reshape((-1, np.prod(sig_dims))).T)\
            - np.dot(dark_image.flatten(), corrected_masks.reshape((-1, np.prod(sig_dims))).T)

        _check_result(
            data=correct_dot, corrected=reconstructed_dot,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_mask_correction_sparse():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        data = _make_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=0)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        masks = sparse.DOK(sparse.zeros((20, ) + sig_dims, dtype=np.float64))
        indices = [np.random.randint(low=0, high=s, size=s//2) for s in (20, ) + sig_dims]
        for tup in zip(*indices):
            masks[tup] = 1
        masks = masks.to_coo()

        data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
        damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

        correct_dot = sparse.dot(data_flat, masks.reshape((-1, np.prod(sig_dims))).T)
        corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)
        assert is_sparse(corrected_masks)

        reconstructed_dot =\
            sparse.dot(damaged_flat, corrected_masks.reshape((-1, np.prod(sig_dims))).T)\
            - sparse.dot(dark_image.flatten(), corrected_masks.reshape((-1, np.prod(sig_dims))).T)

        _check_result(
            data=correct_dot, corrected=reconstructed_dot,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_mask_patch():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-basedcorrection is performed as float64 since it creates
        # numerical instabilities otherwise
        data = _make_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=3)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        masks = (np.random.random((2, ) + sig_dims) - 0.3).astype(np.float64)
        data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
        damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

        correct_dot = np.dot(data_flat, masks.reshape((-1, np.prod(sig_dims))).T)
        corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)

        assert not is_sparse(corrected_masks)

        reconstructed_dot =\
            np.dot(damaged_flat, corrected_masks.reshape((-1, np.prod(sig_dims))).T)\
            - np.dot(dark_image.flatten(), corrected_masks.reshape((-1, np.prod(sig_dims))).T)

        _check_result(
            data=correct_dot, corrected=reconstructed_dot,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_mask_patch_sparse():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        data = _make_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=3)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        masks = sparse.DOK(sparse.zeros((20, ) + sig_dims, dtype=np.float64))
        indices = [np.random.randint(low=0, high=s, size=s//2) for s in (20, ) + sig_dims]
        for tup in zip(*indices):
            masks[tup] = 1
        masks = masks.to_coo()

        data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
        damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

        correct_dot = sparse.dot(data_flat, masks.reshape((-1, np.prod(sig_dims))).T)
        corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)
        assert is_sparse(corrected_masks)

        reconstructed_dot =\
            sparse.dot(damaged_flat, corrected_masks.reshape((-1, np.prod(sig_dims))).T)\
            - sparse.dot(dark_image.flatten(), corrected_masks.reshape((-1, np.prod(sig_dims))).T)

        _check_result(
            data=correct_dot, corrected=reconstructed_dot,
            atol=1e-8, rtol=1e-5
        )


@pytest.mark.with_numba
def test_mask_patch_overlapping():
    for i in range(10):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        # Constant data to reconstruct neighboring damaged pixels faithfully
        data = np.ones(nav_dims + sig_dims, dtype=np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        # Neighboring excluded pixels
        exclude = np.ones((num_sig_dims, 3), dtype=np.int32)
        exclude[0, 1] += 1
        exclude[1, 2] += 1

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        masks = (np.random.random((2, ) + sig_dims) - 0.5).astype(np.float64)
        data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
        damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

        correct_dot = np.dot(data_flat, masks.reshape((-1, np.prod(sig_dims))).T)
        corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)

        assert not is_sparse(corrected_masks)

        reconstructed_dot =\
            np.dot(damaged_flat, corrected_masks.reshape((-1, np.prod(sig_dims))).T)\
            - np.dot(dark_image.flatten(), corrected_masks.reshape((-1, np.prod(sig_dims))).T)

        _check_result(
            data=correct_dot, corrected=reconstructed_dot,
            atol=1e-8, rtol=1e-5
        )
