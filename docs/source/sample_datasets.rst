.. _`sample data`:

================
Sample Datasets
================

Bullseye and circular probe diffraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scanning convergent beam electron diffraction data of gold nanoparticles
(:code:`4DSTEM_experiment/data/datacubes/polyAu_4DSTEM/data`) and simulated
strained gold (:code:`4DSTEM_experiment/data/datacubes/simulation_4DSTEM/data`)
with one file using a standard circular aperture and another using a bullseye
aperture.

:Link:
    https://zenodo.org/record/3592520 :cite:`ophus_colin_2019_3592520,Zeltmann2019`
:Format:
    HDF5 (uint16)
:Dimension:
    4D (100, 84, 250, 250)
:Size:
    2.1 GB

Electron Bessel beam diffraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scanning convergent beam electron diffraction with ring-shaped aperture and
overlapping diffraction orders.

:Link:
    https://zenodo.org/record/2566137 :cite:`giulio_guzzinati_2019_2566137,Guzzinati2019`
:Format:
    Stack of DM3 (currently only scripting)
:Dimension:
    3D
:Size:
    2.6 GB

.. _`hires STO`:

High-resolution 4D STEM dataset of SrTiO3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset can be used to test various analysis methods for high-resolution 4D
STEM, including phase contrast methods such as ptychography.

:Link:
    https://zenodo.org/record/5113449 :cite:`strauch_achim_2021_5113449`
:Format:
    MIB (6 bit)
:Dimension:
    4D (128, 128, 256, 256)
:Size:
    177 MB

.. _`synthetic STO`:

Synthetic 4D STEM dataset based on a SrTiO3 supercell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset allows to investigate phase contrast methods for 4D scanning
transmission electron microscopy, such as ptychography.

:Link:
    https://zenodo.org/record/5113235 :cite:`strauch_achim_2021_5113235`
:Format:
    RAW (float32)
:Dimension:
    4D (100, 100, 596, 596)
:Size:
    14.2 GB

Creating random data
~~~~~~~~~~~~~~~~~~~~~~~

Random data can be generated in the following way. It should be kept in mind
that the data generated in this way can only be used for simple testing as it
has no physical significance.

**Raw file:**

.. testsetup:: sampledataraw

    import os
    import tempfile
    raw_temp = tempfile.TemporaryDirectory()
    os.chdir(raw_temp.name)

.. testcode:: sampledataraw

    # Create sample raw file
    import numpy as np
    sample_data = np.random.randn(16, 16, 16, 16).astype("float32")
    sample_data.tofile("raw_sample.raw")

.. testcode:: sampledataraw

    # Load through Python API
    from libertem.api import Context
    if __name__ == '__main__':
      ctx = Context()
      ds = ctx.load("raw", path="./raw_sample.raw", nav_shape=(16, 16), dtype="float32", sig_shape=(16, 16))

.. testcleanup:: sampledataraw

    os.chdir("..")
    raw_temp.cleanup()

**HDF5 file:**

.. testsetup:: sampledatahdf5

    import os
    import tempfile
    hdf5_temp = tempfile.TemporaryDirectory()
    os.chdir(hdf5_temp.name)

.. testcode:: sampledatahdf5

    # Create sample HDF5 file
    import h5py
    import numpy as np
    file = h5py.File('hdf5_sample.h5','w')
    sample_data = np.random.randn(16,16,16,16).astype("float32")
    dataset = file.create_dataset("dataset",(16,16,16,16), data=sample_data)
    file.close()

.. testcode:: sampledatahdf5

    # Load through Python API
    from libertem.api import Context
    if __name__ == '__main__':
      ctx = Context()
      ds = ctx.load("hdf5", path="./hdf5_sample.h5", ds_path="/dataset")

.. testcleanup:: sampledatahdf5

    os.chdir("..")
    hdf5_temp.cleanup()

Alternatively, you can enter the parameters (scan_size, dtype, detector_size)
directly into the load dialog of the GUI. For more details on loading, please
check :ref:`loading data`.
