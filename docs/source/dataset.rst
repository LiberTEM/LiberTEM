===============
Sample Dataset
===============
Public datasets
~~~~~~~~~~~~~~~~
Some data to work with have been obtained from _`Zenodo <https://zenodo.org>`. 

+-------------------------------------------------+----------+---------+----------+----------+
|   Title                                         |Download  |  Format | Dimension| Size     |
+=================================================+==========+=========+==========+==========+
| `Bullseye probe`_                               |`link1`_  | HDF5    | 4D       | 2.1 GB   |             
+-------------------------------------------------+----------+---------+----------+----------+
| `Circular probe`_                               |`link2`_  |  HDF5   |  4D      | 2.1 GB   |
+-------------------------------------------------+----------+---------+----------+----------+
| `Electron Bessel beam diffraction pattern`_     |`link3`_  | DM3     | 3D       | 2.6 GB   |
+-------------------------------------------------+----------+---------+----------+----------+
Creating your own dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can generate your own dataset in the following way

.. code-block:: python
      
      import numpy as np
      real_data = np.random.randn(16, 16, 16, 16).astype("float32")
      real_data.tofile("/tmp/real_raw_file.raw")

Now you can load the data through the :ref:`Python API <api>` in the following way

.. code-block:: python
    
    from libertem.api import Context
    ctx = Context()
    ds = ctx.load("raw", path="/tmp/something.raw", scan_size=(16, 16), dtype="float32", detector_size=(16, 16))
    
Alternatively, you can enter the parameters (scan_size, dtype, detector_size) directly into the load dialog of the GUI. 

.. _link1: https://zenodo.org/record/3592520/files/calibrationData_bullseyeProbe.h5?download=1
.. _link2: https://zenodo.org/record/3592520/files/calibrationData_circularProbe.h5?download=1
.. _link3: https://zenodo.org/record/2566137/files/experimental_data.7z?download=1
.. _Bullseye probe: https://zenodo.org/record/3592520#.XmdNN3DhXIU
.. _Circular probe: https://zenodo.org/record/3592520#.XmdNN3DhXIU  
.. _Electron Bessel beam diffraction pattern: https://zenodo.org/record/2566137#.XmdNQnDhXIU
