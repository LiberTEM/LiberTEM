================
Sample Datasets
================

Public datasets
~~~~~~~~~~~~~~~~

Some data to work with have been obtained from Zenodo.

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

.. code-block:: python
      
    import numpy as np
    real_data = np.random.randn(16, 16, 16, 16).astype("float32")
    real_data.tofile("/tmp/real_raw_file.raw")

Now you can load the data through the :ref:`api documentation` in the following way:

.. code-block:: python
    
    from libertem.api import Context
    ctx = Context()
    ds = ctx.load("raw", path="/tmp/real_raw_file.raw", scan_size=(16, 16), dtype="float32", detector_size=(16, 16))
    
Alternatively, you can enter the parameters (scan_size, dtype, detector_size)
directly into the load dialog of the GUI.

.. _`Bullseye and circular probe diffraction`: https://zenodo.org/record/3592520
.. _`Electron Bessel beam diffraction`: https://zenodo.org/record/2566137
