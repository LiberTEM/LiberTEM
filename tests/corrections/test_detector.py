import numpy as np
import sparse

from libertem.corrections import detector
from libertem.masks import is_sparse


def test_detector_correction():
    for i in range(10):
        num_nav_dims = np.random.choice([1, 2, 3])
        num_sig_dims = np.random.choice([1, 2, 3])

        nav_dims = tuple(np.random.randint(low=1, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=1, high=16, size=num_sig_dims))

        data = np.arange(np.prod(nav_dims) * np.prod(sig_dims), dtype=np.float32) + 1
        data = data.reshape(nav_dims + sig_dims)

        gain_map = np.random.random(sig_dims) + 1
        dark_image = np.random.random(sig_dims) * 0.01

        num_excluded = np.random.choice([0, 1, 2])
        exclude = np.array([np.random.randint(low=0, high=s, size=num_excluded) for s in sig_dims])

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 0

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        try:
            corrected = detector.correct(
                buffer=damaged_data,
                dark_image=dark_image,
                gain_map=gain_map,
                excluded_pixels=exclude,
                inplace=False
            )
        except detector.RepairValueError as e:
            print(e)
            continue

        atol = 5
        rtol = 0.01

        m1 = np.unravel_index(np.argmax(np.abs(data - corrected)), nav_dims + sig_dims)
        m2 = np.unravel_index(
            np.argmax(np.abs(data - corrected) / np.abs(data)), nav_dims + sig_dims
        )
        # From np.allclose documentation
        select = np.abs(data - corrected) > (atol + rtol * np.abs(data))
        print("Maximum absolute error: data %s, corrected %s" % (data[m1], corrected[m1]))
        print("Maximum relative error: data %s, corrected %s, relative %s" %
            (data[m2], corrected[m2], np.abs(data[m2] - corrected[m2]) / np.abs(data[m2]))
        )
        print("Triggering tolerance limits: data %s, corrected %s, relative %s" %
            (data[select],
            corrected[select],
            np.abs(data[select] - corrected[select]) / np.abs(data[select]))
        )
        # Make sure we didn't do it in place
        assert not np.allclose(corrected, damaged_data)

        # Realistic chance to correct a missing pixel with enough neighbors
        if np.prod(sig_dims) > 16 and min(sig_dims) >= 3:
            # allclose() is too sensitive to occasional outliers
            assert np.linalg.norm(data - corrected) / np.linalg.norm(data) < 0.05

        # If we didn't patch dead pixels, the result should be precise
        if num_excluded == 0:
            assert np.allclose(data, corrected)

        detector.correct(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            excluded_pixels=exclude,
            inplace=True
        )

        # Now damaged_data should be modified and equal to corrected
        assert np.allclose(corrected, damaged_data)


def test_detector_patch():
    for i in range(10):
        num_nav_dims = np.random.choice([2, 3])
        num_sig_dims = np.random.choice([2, 3])

        nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
        sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

        data = np.ones(nav_dims + sig_dims, dtype=np.float32)

        gain_map = np.random.random(sig_dims) + 1
        dark_image = np.random.random(sig_dims) * 0.01

        num_excluded = 3
        exclude = np.array([np.random.randint(low=0, high=s, size=num_excluded) for s in sig_dims])

        damaged_data = data.copy()
        damaged_data /= gain_map
        damaged_data += dark_image
        damaged_data[(Ellipsis, *exclude)] = 0

        print("Nav dims: ", nav_dims)
        print("Sig dims:", sig_dims)
        print("Exclude: ", exclude)

        try:
            corrected = detector.correct(
                buffer=damaged_data,
                dark_image=dark_image,
                gain_map=gain_map,
                excluded_pixels=exclude,
                inplace=False
            )
        except detector.RepairValueError as e:
            print(e)
            continue

        atol = 1e-8
        rtol = 1e-5

        m1 = np.unravel_index(np.argmax(np.abs(data - corrected)), nav_dims + sig_dims)
        m2 = np.unravel_index(
            np.argmax(np.abs(data - corrected) / np.abs(data)), nav_dims + sig_dims
        )
        # From np.allclose documentation
        select = np.abs(data - corrected) > (atol + rtol * np.abs(data))
        print("Maximum absolute error: data %s, corrected %s" % (data[m1], corrected[m1]))
        print("Maximum relative error: data %s, corrected %s, relative %s" %
            (data[m2], corrected[m2], np.abs(data[m2] - corrected[m2]) / np.abs(data[m2]))
        )
        print("Triggering tolerance limits: data %s, corrected %s, relative %s" %
            (data[select],
            corrected[select],
            np.abs(data[select] - corrected[select]) / np.abs(data[select]))
        )
        # The settings were chosen to make patching the pixel easy
        assert np.allclose(data, corrected, atol=atol, rtol=rtol)


def test_mask_correction():
    num_nav_dims = np.random.choice([1, 2, 3])
    num_sig_dims = np.random.choice([2, 3])

    nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
    sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

    # We choose data that is easy to patch
    data = data = np.ones(nav_dims + sig_dims, dtype=np.float32)

    gain_map = np.random.random(sig_dims) + 1
    dark_image = np.random.random(sig_dims) * 0.01

    num_excluded = 3
    exclude = np.array([np.random.randint(low=0, high=s, size=num_excluded) for s in sig_dims])

    damaged_data = data.copy()
    damaged_data /= gain_map
    damaged_data += dark_image
    damaged_data[(Ellipsis, *exclude)] = 0

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)
    print("Exclude: ", exclude)

    masks = np.random.random((2, ) + sig_dims).reshape((-1, np.prod(sig_dims))) - 0.5
    data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
    damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

    correct_dot = np.dot(data_flat, masks.T)
    corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)

    assert not is_sparse(corrected_masks)

    reconstructed_dot =\
        np.dot(damaged_flat, corrected_masks.T)\
        - np.dot(dark_image.flatten(), corrected_masks.T)

    assert np.allclose(correct_dot, reconstructed_dot)


def test_mask_correction_sparse():
    return
    np.random.random()
    num_nav_dims = np.random.choice([1, 2, 3])
    num_sig_dims = np.random.choice([2, 3])

    nav_dims = tuple(np.random.randint(low=8, high=16, size=num_nav_dims))
    sig_dims = tuple(np.random.randint(low=8, high=16, size=num_sig_dims))

    # We choose data that is easy to patch
    data = np.ones(nav_dims + sig_dims, dtype=np.float32)

    gain_map = np.random.random(sig_dims) + 1
    dark_image = np.random.random(sig_dims) * 0.01

    num_excluded = 3
    exclude = np.array([np.random.randint(low=0, high=s, size=num_excluded) for s in sig_dims])

    damaged_data = data.copy()
    damaged_data /= gain_map
    damaged_data += dark_image
    damaged_data[(Ellipsis, *exclude)] = 0

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)
    print("Exclude: ", exclude)

    masks = sparse.zeros((20, ) + sig_dims)
    indices = np.array([np.random.randint(low=0, high=s, size=s//2) for s in (20, ) + sig_dims])
    masks[(*indices, )] = 1

    data_flat = data.reshape((np.prod(nav_dims), np.prod(sig_dims)))
    damaged_flat = damaged_data.reshape((np.prod(nav_dims), np.prod(sig_dims)))

    correct_dot = sparse.dot(data_flat, masks.reshape((-1, np.prod(sig_dims))).T)
    corrected_masks = detector.correct_dot_masks(masks, gain_map, exclude)
    assert is_sparse(corrected_masks)

    reconstructed_dot =\
        sparse.dot(damaged_flat, corrected_masks.reshape((-1, np.prod(sig_dims))).T)\
        - sparse.dot(dark_image.flatten(), corrected_masks.reshape((-1, np.prod(sig_dims))).T)

    assert np.allclose(correct_dot, reconstructed_dot)
