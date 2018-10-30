|gitter|_ |travis|_ |appveyor|_

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _gitter: https://gitter.im/LiberTEM/Lobby

.. |travis| image:: https://api.travis-ci.org/LiberTEM/LiberTEM.svg?branch=master
.. _travis: https://travis-ci.org/LiberTEM/LiberTEM

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/wokeo6ee2frq481m?svg=true
.. _appveyor: https://ci.appveyor.com/project/sk1p/libertem



LiberTEM is an open source platform for high-throughput distributed processing of `pixelated <https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy#Universal_detectors>`_ scanning transmission electron microscopy (`STEM <https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy>`_) data.

It is designed for high throughput and scalability on PCs, single server nodes, clusters and cloud services. On clusters it can use the Hadoop file system with fast distributed
local storage on high-performance SSDs. That way it achieves very high collective IO performance on a compact and cost-efficient system built from stock components.
With cached file system reads it can reach a throughput of up to 14 GB/s per processing node with a quad-core CPU.

LiberTEM is supported on Linux, Mac OS X and Windows. Other platforms
that allow installation of Python 3 and the required packages will likely work as well. The GUI is running
in a web browser.

LiberTEM currently opens most file formats used for pixelated STEM:

- Raw binary files, for example for the Thermo Fisher EMPAD detector
- Quantum Detectors MIB format (currently alpha, more testing and sample files highly appreciated)
- Nanomegas .blo block files
- Gatan K2IS raw format (currently beta)
- HDF5-based formats such as Hyperspy files, NeXus and EMD
- Please contact us if you are interested in support for an additional format!

The 0.1 release implements anything that can be done by applying masks to each detector frame,
for example the numerous virtual detector methods (virtual bright field, virtual HAADF, ...) or center of mass. 

The GUI of the 0.1 release allows interactive analysis and data exploration with basic virtual
detectors, center of mass and display of individual detector frames.

The Python API can be used for more complex operations with arbitrary masks and other features like data export. There are example Jupyter notebooks available in the `examples directory <https://github.com/LiberTEM/LiberTEM/blob/master/examples>`_.
If you are having trouble running the examples, please let us know, either by filing an issue
or by `joining our Gitter chat <https://gitter.im/LiberTEM/Lobby>`_.

LiberTEM is suitable as a high-performance processing backend for other applications, including live data streams. `Contact us <https://gitter.im/LiberTEM/Lobby>`_ if you are interested! 

Deployment as a single-node system for a local user is thoroughly tested and can be considered stable in the 0.1 release. Deployment on a cluster is 
experimental and still requires some additional work, see `Issue #105 <https://github.com/LiberTEM/LiberTEM/issues/105>`_.

LiberTEM is evolving rapidly and prioritizes features following user demand and contributions. In the future we'd like to implement live acquisition, and more analysis methods for all applications of pixelated STEM.
If you like to influence the direction this
project is taking, or if you'd like to contribute, please join our `gitter chat <https://gitter.im/LiberTEM/Lobby>`_,
our `development mailing list <https://groups.google.com/forum/#!forum/libertem-dev>`_,
and our `general mailing list <https://groups.google.com/forum/#!forum/libertem>`_. 

Installation
------------

The short version:

.. code-block:: shell

    $ virtualenv -p python3.6 ~/libertem-venv/
    $ source ~/libertem-venv/bin/activate
    (libertem) $ pip install libertem[torch]

Please see `our documentation <https://libertem.github.io/LiberTEM/install.html>`_ for details!
