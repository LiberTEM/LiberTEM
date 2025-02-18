import numpy as np
import sparse
import pytest

from libertem.utils.generate import gradient_data, exclude_pixels
from libertem.io.corrections import detector
from libertem.common.sparse import is_sparse


REPEATS = 1


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
    print(f"Maximum absolute error: data {data[m1]}, corrected {corrected[m1]}")
    print("Maximum relative error: data %s, corrected %s, relative %s" %
        (data[m2], corrected[m2], relative_error[m2])
    )
    print("Triggering tolerance limits: data %s, corrected %s, relative %s" %
        (data[select],
        corrected[select],
        relative_error[select])
    )

    assert np.allclose(data, corrected, atol=atol, rtol=rtol)


@pytest.mark.with_numba
def test_detector_correction():
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([1, 2, 3])

        nav_dims = tuple(np.random.randint(low=1, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=1, high=16, size=num_sig_dims))

        data = gradient_data(nav_dims, sig_dims)

        # Test pure gain and offset correction without
        # patching pixels
        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=0)

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
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([1, 2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        data = np.ones(nav_dims + sig_dims, dtype=np.uint8)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=2)

        gain_map = np.ones(sig_dims)
        dark_image = (np.random.random(sig_dims) * 3).astype(np.uint8)
        # Make sure the dark image is not all zero so that
        # the damaged data is different from the original
        # https://github.com/LiberTEM/LiberTEM/issues/910
        # This is only necessary for an integer dark image
        # since for float it would be extremely unlikely
        # that all values are exactly 0
        atleastone = np.random.randint(0, np.prod(sig_dims))
        dark_image[np.unravel_index(atleastone, sig_dims)] = 1

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
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        data = gradient_data(nav_dims, sig_dims)

        gain_map = np.random.random(sig_dims) + 1
        dark_image = np.random.random(sig_dims)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=3)

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


def test_detector_patch_large():
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = 2

        nav_dims = tuple(np.random.randint(low=3, high=5, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=4*32, high=1024, size=num_sig_dims))

        data = gradient_data(nav_dims, sig_dims)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=999)

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


def test_detector_patch_too_large():
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = 2

        nav_dims = tuple(np.random.randint(low=3, high=5, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=4*32, high=1024, size=num_sig_dims))

        data = gradient_data(nav_dims, sig_dims)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=1001)

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
@pytest.mark.slow
def test_detector_patch_overlapping():
    for i in range(REPEATS):
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
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        data = gradient_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=0)

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
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        data = gradient_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=0)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        shape = (20, ) + sig_dims
        count = min(shape)//2
        assert count > 0
        indices = np.stack([np.random.randint(low=0, high=s, size=count) for s in shape], axis=0)
        masks = sparse.COO(coords=indices, shape=shape, data=1).astype(np.float64)

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
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-basedcorrection is performed as float64 since it creates
        # numerical instabilities otherwise
        data = gradient_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=3)

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
    for i in range(REPEATS):
        print(f"Loop number {i}")
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        # The mask-based correction is performed as float64 since it creates
        # numerical instabilities otherwise
        data = gradient_data(nav_dims, sig_dims).astype(np.float64)

        gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
        dark_image = np.random.random(sig_dims).astype(np.float64)

        exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=3)

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 1e24

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        shape = (20, ) + sig_dims
        count = min(shape)//2
        assert count > 0
        indices = np.stack([np.random.randint(low=0, high=s, size=count) for s in shape], axis=0)
        masks = sparse.COO(coords=indices, shape=shape, data=1).astype(np.float64)

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
    for i in range(REPEATS):
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
