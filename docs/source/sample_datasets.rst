.. _`sample data`:

================
Sample Datasets
================

Public datasets
~~~~~~~~~~~~~~~~

Some data to work with can be downloaded from Zenodo.

+-----------------------------------------------------+-------------------------------------------------------------------+-----------------+---------------------+--------+
| Link                                                | Description                                                       | Format          | Dimension           | Size   |
+=====================================================+===================================================================+=================+=====================+========+
| `Bullseye and circular probe diffraction`_          | Scanning convergent beam electron diffraction data                | HDF5 (uint16)   | 4D                  | 2.1 GB |
| :cite:`ophus_colin_2019_3592520,Zeltmann2019`       | of gold nanoparticles                                             |                 |                     |        |
|                                                     | (:code:`4DSTEM_experiment/data/datacubes/polyAu_4DSTEM/data`)     |                 | (100, 84, 250, 250) |        |
|                                                     | and simulated strained gold                                       |                 |                     |        |
|                                                     | (:code:`4DSTEM_experiment/data/datacubes/simulation_4DSTEM/data`) |                 |                     |        |
|                                                     | with one file using a standard circular aperture and another      |                 |                     |        |
|                                                     | using a bullseye aperture.                                        |                 |                     |        |
+-----------------------------------------------------+-------------------------------------------------------------------+-----------------+---------------------+--------+
| `Electron Bessel beam diffraction`_                 | Scanning convergent beam electron diffraction with ring-shaped    | Stack of DM3    | 3D                  | 2.6 GB |
| :cite:`giulio_guzzinati_2019_2566137,Guzzinati2019` | aperture and overlapping diffraction orders.                      | (currently only |                     |        |
|                                                     |                                                                   | scripting)      |                     |        |
+-----------------------------------------------------+-------------------------------------------------------------------+-----------------+---------------------+--------+

Creating random data
~~~~~~~~~~~~~~~~~~~~~~~

Random data can be generated in the following way. It should be kept in mind
that the data generated in this way can only be used for simple testing as it
has no physical significance.

**Raw file:**

.. testsetup:: sampledataraw

    import os
    os.mkdir("temp_sample_raw")
    os.chdir("temp_sample_raw")

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
      ds = ctx.load("raw", path="./raw_sample.raw", scan_size=(16, 16), dtype="float32", detector_size=(16, 16))

.. testcleanup:: sampledataraw

    import shutil
    os.chdir("..")
    shutil.rmtree("temp_sample_raw")

**HDF5 file:**

.. testsetup:: sampledatahdf5

    import os
    os.mkdir("temp_sample_hdf5")
    os.chdir("temp_sample_hdf5")

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

    import shutil
    os.chdir("..")
    shutil.rmtree("temp_sample_hdf5")

Alternatively, you can enter the parameters (scan_size, dtype, detector_size)
directly into the load dialog of the GUI. For more details on loading, please
check :ref:`loading data`.

.. _`Bullseye and circular probe diffraction`: https://zenodo.org/record/3592520
.. _`Electron Bessel beam diffraction`: https://zenodo.org/record/2566137
