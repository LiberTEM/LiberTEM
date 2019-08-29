import numpy as np
from libertem.utils import make_cartesian, make_polar, frame_peaks
import libertem.masks as m


def cbed_frame(fy=128, fx=128, zero=None, a=None, b=None, indices=None, radius=4):
    if zero is None:
        zero = (fy//2, fx//2)
    zero = np.array(zero)
    if a is None:
        a = (fy//8, 0)
    a = np.array(a)
    if b is None:
        b = make_cartesian(make_polar(a))
    b = np.array(b)
    if indices is None:
        indices = np.mgrid[-10:11, -10:11]
    indices, peaks = frame_peaks(fy=fy, fx=fx, zero=zero, a=a, b=b, r=radius, indices=indices)

    data = np.zeros((1, fy, fx), dtype=np.float32)

    for p in peaks:
        data += m.circular(
            centerX=p[1],
            centerY=p[0],
            imageSizeX=fx,
            imageSizeY=fy,
            radius=radius,
            antialiased=True,
        )

    return (data, indices, peaks)
