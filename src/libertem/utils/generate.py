import numpy as np
from libertem.utils import make_cartesian, make_polar, frame_peaks
import libertem.masks as m


def cbed_frame(fy=128, fx=128, zero=None, a=None, b=None, indices=None, radius=4, all_equal=False):
    if zero is None:
        zero = (fy//2, fx//2)
    zero = np.array(zero)
    if a is None:
        a = (fy//8, 0)
    a = np.array(a)
    if b is None:
        b = make_cartesian(make_polar(a) - (0, np.pi/2))
    b = np.array(b)
    if indices is None:
        indices = np.mgrid[-10:11, -10:11]
    indices, peaks = frame_peaks(fy=fy, fx=fx, zero=zero, a=a, b=b, r=radius, indices=indices)

    data = np.zeros((1, fy, fx), dtype=np.float32)

    dists = np.linalg.norm(peaks - zero, axis=-1)
    max_dist = dists.max()

    for i, p in enumerate(peaks):
        data += m.circular(
            centerX=p[1],
            centerY=p[0],
            imageSizeX=fx,
            imageSizeY=fy,
            radius=radius,
            antialiased=True,
        ) * (1 if all_equal else max_dist - dists[i] + i)

    return (data, indices, peaks)
