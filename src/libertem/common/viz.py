from PIL import Image
from io import BytesIO


def encode_image(result, save_kwargs=None):
    """
    Save the RGBA data in ``result`` to an image with parameters ``save_kwargs``
    passed to ``PIL.Image.save``.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    save_kwargs : dict or None
        dict of kwargs passed to Pillow when saving the image, can be used to set
        the file format, quality, ...

    Returns
    -------

    BytesIO
        a buffer containing the result image (as PNG/JPG/... depending on save_kwargs)
    """
    if save_kwargs is None:
        save_kwargs = {'format': 'png'}
    # see also: https://stackoverflow.com/a/10967471/540644
    im = Image.fromarray(result)
    buf = BytesIO()
    im = im.convert(mode="RGB")
    im.save(buf, **save_kwargs)
    buf.seek(0)
    return buf
