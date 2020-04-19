|gitter|_ |travis|_ |appveyor|_ |zenodo|_ |github|_ |codeclimate|_

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _gitter: https://gitter.im/LiberTEM/Lobby

.. |travis| image:: https://api.travis-ci.org/LiberTEM/LiberTEM.svg?branch=master
.. _travis: https://travis-ci.org/LiberTEM/LiberTEM

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/wokeo6ee2frq481m/branch/master?svg=true
.. _appveyor: https://ci.appveyor.com/project/sk1p/libertem

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1477847.svg
.. _zenodo: https://doi.org/10.5281/zenodo.1477847

.. |github| image:: https://img.shields.io/badge/GitHub-GPL--3.0-informational
.. _github: https://github.com/LiberTEM/LiberTEM/


.. |codeclimate| image:: https://api.codeclimate.com/v1/badges/dee042f64380f64737e5/maintainability
.. _codeclimate: https://codeclimate.com/github/LiberTEM/LiberTEM

LiberTEM is an open source platform for high-throughput distributed processing
of large-scale binary data sets using a simplified `MapReduce programming model
<https://en.wikipedia.org/wiki/MapReduce>`_. The current focus is `pixelated
<https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy#Universal_detectors_(4D_STEM)>`_
scanning transmission electron microscopy (`STEM
<https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy>`_)
:cite:`doi:10.1002/9783527808465.EMC2016.6284,Ophus_2019` and scanning electron beam
diffraction data.

It is `designed for high throughput and scalability
<https://libertem.github.io/LiberTEM/architecture.html>`_ on PCs, single server
nodes, clusters and cloud services. On clusters it can use fast distributed
local storage on high-performance SSDs. That way it achieves `very high
aggregate IO performance
<https://libertem.github.io/LiberTEM/performance.html>`_ on a compact and
cost-efficient system built from stock components.

LiberTEM is supported on Linux, Mac OS X and Windows. Other platforms that allow
installation of Python 3 and the required packages will likely work as well. The
GUI is running in a web browser.

Installation
------------

The short version:

.. code-block:: shell

    $ virtualenv -p python3 ~/libertem-venv/
    $ source ~/libertem-venv/bin/activate
    (libertem) $ pip install "libertem[torch]"

Please see `our documentation <https://libertem.github.io/LiberTEM/install.html>`_ for details!

Deployment as a single-node system for a local user is thoroughly tested and can be considered stable. Deployment on a cluster is
experimental and still requires some additional work, see `Issue #105 <https://github.com/LiberTEM/LiberTEM/issues/105>`_.

Applications
------------

- Virtual detectors (virtual bright field, virtual HAADF, center of mass :cite:`Krajnak2016`,
  custom shapes via masks)
- `Analysis of amorphous materials <https://libertem.github.io/LiberTEM/app/amorphous.html>`_
- `Strain mapping <https://libertem.github.io/LiberTEM-blobfinder/>`_
- `Custom analysis functions (user-defined functions) <https://libertem.github.io/LiberTEM/udf.html>`_
- `Off-axis electron holography reconstruction <https://libertem.github.io/LiberTEM/app/holography.html>`_

Please see `the applications section <https://libertem.github.io/LiberTEM/applications.html>`_ of our documentation for details!

The Python API and user-defined functions (UDFs) can be used for more complex operations with arbitrary masks and other features like data export. There are example Jupyter notebooks available in the `examples directory <https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_.
If you are having trouble running the examples, please let us know, either by filing an issue
or by `joining our Gitter chat <https://gitter.im/LiberTEM/Lobby>`_.

LiberTEM is suitable as a high-performance processing backend for other applications, including live data streams. `Contact us <https://gitter.im/LiberTEM/Lobby>`_ if you are interested!


LiberTEM is evolving rapidly and prioritizes features following user demand and contributions. In the future we'd like to implement live acquisition, and more analysis methods for all applications of pixelated STEM and other large-scale detector data.
If you like to influence the direction this
project is taking, or if you'd like to `contribute <https://libertem.github.io/LiberTEM/contributing.html>`_, please join our `gitter chat <https://gitter.im/LiberTEM/Lobby>`_
and our `general mailing list <https://groups.google.com/forum/#!forum/libertem>`_.

File formats
------------

LiberTEM currently opens most file formats used for pixelated STEM. See `our general information on loading data <https://libertem.github.io/LiberTEM/formats.html>`_
and `format-specific documentation <https://libertem.github.io/LiberTEM/reference/dataset.html#formats>`_ for more information!

- Raw binary files
- Thermo Fisher EMPAD detector :cite:`Tate2016` files
- `Quantum Detectors MIB format <http://quantumdetectors.com/wp-content/uploads/2017/01/1532-Merlin-for-EM-Technical-Datasheet-v2.pdf>`_
- Nanomegas .blo block files
- Direct Electron DE5 files (HDF5-based) for `DE-Series <http://www.directelectron.com/de-series/>`_ detectors 
- `Gatan K2 IS <https://web.archive.org/web/20180809021832/http://www.gatan.com/products/tem-imaging-spectroscopy/k2-camera>`_ raw format
- Stacks of Gatan DM3 and DM4 files (via `openNCEM <https://github.com/ercius/openNCEM>`_)
- FRMS6 from PNDetector pnCCD cameras :cite:`Simson2015` (currently alpha, gain correction still needs UI changes)
- FEI SER files (via `openNCEM <https://github.com/ercius/openNCEM>`_)
- HDF5-based formats such as Hyperspy files, NeXus and EMD
- Please contact us if you are interested in support for an additional format!

License
-------

LiberTEM is licensed under GPLv3. The I/O parts are also available under the MIT license, please see LICENSE files in the subdirectories for details.
