import processing
import numpy as np

stackheight = 8
ds = processing.BinaryHDFSDataSet(index_path="test/index.json", host='localhost', port=8020,
                                  stackheight=stackheight)


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    """
    Make a circular mask in a bool array for masking a region in an image.

    Parameters
    ----------
    centreX, centreY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Radius of the mask.

    Returns
    -------
    Boolean Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.

    Examples
    --------
    >>> import numpy as np
    >>> import fpd_data_processing.pixelated_stem_tools as pst
    >>> image = np.ones((9, 9))
    >>> mask = pst._make_circular_mask(4, 4, 9, 9, 2)
    >>> image_masked = image*mask
    >>> import matplotlib.pyplot as plt
    >>> cax = plt.imshow(image_masked)
    """
    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    mask = x*x + y*y <= radius*radius
    return(mask)


job = processing.ApplyMasksJob(dataset=ds, masks=[
    np.random.randn(128, 128),
    np.ones(shape=(128, 128)) * _make_circular_mask(
        centerX=64, centerY=64,
        imageSizeX=128, imageSizeY=128, radius=25
    ),
])

executor = processing.DaskJobExecutor(scheduler_uri="tcp://localhost:8786")
print(executor.client.scheduler_info())

result = np.zeros(shape=(job.maskcount, 256, 256))
for res in executor.run_job(job):
    for tile in res:
        tile.copy_to_result(result)
