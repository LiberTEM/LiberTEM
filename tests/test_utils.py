import pytest
import numpy as np
from scipy.ndimage import gaussian_filter
import libertem.utils as ut
from libertem.utils.generate import hologram_frame


def test_polar():
    data = np.array([
        [(0, 1), (0, 1)],
        [(1, 0), (2, 0)],
        [(-2, 0), (0, 1)],
    ])
    expected = np.array([
        [(1, 0), (1, 0)],
        [(1, np.pi/2), (2, np.pi/2)],
        [(2, -np.pi/2), (1, 0)],
    ])

    result = ut.make_polar(data)
    assert data.shape == expected.shape
    assert result.shape == expected.shape
    assert np.allclose(expected, result)


def test_conversion(points):
    assert np.allclose(points, ut.make_cartesian(ut.make_polar(points)))


def test_rotation():
    y, x = -1, 2
    r_y, r_x = ut.rotate_deg(y, x, 90)
    # Clock-wise with y pointing down
    assert np.allclose(r_y, 2)
    assert np.allclose(r_x, 1)


def test_rotate_multi():
    y, x = np.random.random(2) - 0.5
    angle = 30
    cos_angle = np.cos(np.pi*angle/180)
    sin_angle = np.sin(np.pi*angle/180)
    assert 360 % angle == 0
    r_y = y
    r_x = x
    for i in range(360//angle):
        r_y2, r_x2 = ut.rotate_deg(
            y=r_y,
            x=r_x,
            degrees=angle
        )
        r_y, r_x = ut.rotate_precalc(
            y=r_y,
            x=r_x,
            cos_angle=cos_angle,
            sin_angle=sin_angle
        )
        assert np.allclose(r_y, r_y2)
        assert np.allclose(r_x, r_x2)
    print(r_y, r_x)
    assert np.allclose(r_y, y)
    assert np.allclose(r_x, x)


def test_polar_rotate():
    y, x, radians = np.random.random(3)
    ((r, phi),) = ut.make_polar(np.array([(y, x)]))
    ((r_y, r_x),) = ut.make_cartesian(np.array([(r, phi + radians)]))
    r_y2, r_x2 = ut.rotate_rad(y, x, radians)
    assert np.allclose(r_y, r_y2)
    assert np.allclose(r_x, r_x2)


@pytest.mark.parametrize('counts, sampling, visibility, f_angle, gauss, poisson, rtol1, rtol2',
                         [(None, None, None, None, None, None, 1e-5, 2e-2),
                          (500., 6.2, 0.5, 66., 0.7, 1e-4, 6e-2, .3)])
def test_hologram_frame(counts, sampling, visibility, f_angle, gauss, poisson, rtol1, rtol2):
    sx, sy = (32, 64)
    x, y = np.meshgrid(np.arange(sx), np.arange(sy))

    kwargs = {}
    if counts:
        kwargs['counts'] = counts
    else:
        counts = 1000.
    if sampling:
        kwargs['sampling'] = sampling
    else:
        sampling = 5.
    if visibility:
        kwargs['visibility'] = visibility
    else:
        visibility = 1.
    if f_angle:
        kwargs['f_angle'] = f_angle
    else:
        f_angle = 30.
    if gauss:
        kwargs['gaussian_noise'] = gauss
    if poisson:
        kwargs['poisson_noise'] = poisson
    amp = 1 - np.random.randn(sy, sx) * 0.01
    phase = np.random.random() * x / sx + np.random.random() * y / sy
    holo_test = hologram_frame(amp, phase, **kwargs)

    f_angle = f_angle / 180. * np.pi

    holo = counts / 2 * (1. + amp ** 2 + 2. * amp * visibility
                         * np.cos(2. * np.pi * y / sampling * np.cos(f_angle)
                                  + 2. * np.pi * x / sampling * np.sin(f_angle)
                                  - phase))

    if poisson:
        noise_scale = poisson * counts
        holo = noise_scale * np.random.poisson(holo / noise_scale)

    if gauss:
        holo = gaussian_filter(holo, gauss)
    assert np.allclose(holo, holo_test, rtol=rtol1)

    # test derived parameters:
    # test if mean value equals to counts
    assert np.isclose(holo_test.mean(), counts, rtol=5e-3)

    # test if calculated contrast is equals to teh input
    def contrast(a):
        return (a.max(1).mean() - a.min(1).mean()) \
                / (a.min(1).mean() + a.max(1).mean())
    assert np.isclose(contrast(holo_test), visibility, rtol=rtol2)

    # test if fringe spacing equals to the input
    holo_fft = np.abs(np.fft.rfft2(holo_test[:sx, :sx]))
    holo_fft[:1, :1] = 0.
    holo_max = np.unravel_index(holo_fft.argmax(), holo_fft.shape)
    holo_max = np.hypot(holo_max[0], holo_max[1])
    sampling_test = sx / holo_max
    error_sampling = sampling_test * (1. / holo_max)
    assert np.isclose(sampling_test, sampling, atol=error_sampling)


def test_holo_frame_asserts():
    # test asserts:
    with pytest.raises(ValueError):
        hologram_frame(np.ones((7, 5)), np.zeros((5, 7)))
    with pytest.raises(ValueError):
        hologram_frame(np.ones((5, 7)), np.zeros((5, 7)), gaussian_noise='a lot')
    with pytest.raises(ValueError):
        hologram_frame(np.ones((5, 7)), np.zeros((5, 7)), poisson_noise='a bit')
