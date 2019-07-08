|gitter|_ |travis|_ |appveyor|_ |zenodo|_

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _gitter: https://gitter.im/LiberTEM/Lobby

.. |travis| image:: https://api.travis-ci.org/LiberTEM/LiberTEM.svg?branch=master
.. _travis: https://travis-ci.org/LiberTEM/LiberTEM

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/wokeo6ee2frq481m/branch/master?svg=true
.. _appveyor: https://ci.appveyor.com/project/sk1p/libertem

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1478763.svg
.. _zenodo: https://doi.org/10.5281/zenodo.1478763



LiberTEM is an open source platform for high-throughput distributed processing of `pixelated <https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy#Universal_detectors>`_ scanning transmission electron microscopy (`STEM <https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy>`_) data :cite:`doi:10.1002/9783527808465.EMC2016.6284`.

It is designed for high throughput and scalability on PCs, single server nodes, clusters and cloud services. On clusters it can use the `Hadoop file system <http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html>`_ with fast distributed
local storage on high-performance SSDs. That way it achieves very high collective IO performance on a compact and cost-efficient system built from stock components.
With cached file system reads it can reach a throughput of up to 14 GB/s per processing node with a quad-core CPU.

LiberTEM is supported on Linux, Mac OS X and Windows. Other platforms
that allow installation of Python 3 and the required packages will likely work as well. The GUI is running
in a web browser.

Installation
------------

The short version:

.. code-block:: shell

    $ virtualenv -p python3.6 ~/libertem-venv/
    $ source ~/libertem-venv/bin/activate
    (libertem) $ pip install libertem[torch]

Please see `our documentation <https://libertem.github.io/LiberTEM/install.html>`_ for details!

Deployment as a single-node system for a local user is thoroughly tested and can be considered stable in the 0.1 release. Deployment on a cluster is 
experimental and still requires some additional work, see `Issue #105 <https://github.com/LiberTEM/LiberTEM/issues/105>`_.

Applications
------------

- Virtual detectors (virtual bright field, virtual HAADF, center of mass :cite:`Krajnak2016`,
  custom shapes via masks)
- Analysis of amorphous materials
- Strain mapping

Please see `the applications section <https://libertem.github.io/LiberTEM/applications.html>`_ of our documentation for details!

The Python API and user-defined functions (UDFs) can be used for more complex operations with arbitrary masks and other features like data export. There are example Jupyter notebooks available in the `examples directory <https://github.com/LiberTEM/LiberTEM/blob/master/examples>`_.
If you are having trouble running the examples, please let us know, either by filing an issue
or by `joining our Gitter chat <https://gitter.im/LiberTEM/Lobby>`_.

LiberTEM is suitable as a high-performance processing backend for other applications, including live data streams. `Contact us <https://gitter.im/LiberTEM/Lobby>`_ if you are interested! 


LiberTEM is evolving rapidly and prioritizes features following user demand and contributions. In the future we'd like to implement live acquisition, and more analysis methods for all applications of pixelated STEM.
If you like to influence the direction this
project is taking, or if you'd like to contribute, please join our `gitter chat <https://gitter.im/LiberTEM/Lobby>`_,
our `development mailing list <https://groups.google.com/forum/#!forum/libertem-dev>`_,
and our `general mailing list <https://groups.google.com/forum/#!forum/libertem>`_. 

File formats
------------

LiberTEM currently opens most file formats used for pixelated STEM:

- Raw binary files, for example for the Thermo Fisher EMPAD detector :cite:`Tate2016`
- `Quantum Detectors MIB format <http://quantumdetectors.com/wp-content/uploads/2017/01/1532-Merlin-for-EM-Technical-Datasheet-v2.pdf>`_ (currently beta, more testing and sample files still highly appreciated)
- Nanomegas .blo block files
- `Gatan K2 IS <http://www.gatan.com/products/tem-imaging-spectroscopy/k2-camera>`_ raw format
- FRMS6 from PNDetector pnCCD cameras :cite:`Simson2015` (currently alpha, gain correction still needs UI changes)
- FEI SER files (via `openNCEM <https://github.com/ercius/openNCEM>`_)
- HDF5-based formats such as Hyperspy files, NeXus and EMD
- Please contact us if you are interested in support for an additional format!


License
-------

LiberTEM is licensed under GPLv3. The I/O parts are also available under the MIT license, please see LICENSE files in the subdirectories for details.
