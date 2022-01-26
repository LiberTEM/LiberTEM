|docs|_ |gitter|_ |azure|_ |github|_ |codeclimate|_ |precommit|_ |joss|_ |zenodo|_

.. |docs| image:: https://img.shields.io/badge/%F0%9F%95%AE-docs-green.svg
.. _docs: https://libertem.github.io/LiberTEM/

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _gitter: https://gitter.im/LiberTEM/Lobby

.. |azure| image:: https://dev.azure.com/LiberTEM/LiberTEM/_apis/build/status/LiberTEM.LiberTEM?branchName=master
.. _azure: https://dev.azure.com/LiberTEM/LiberTEM/_build/latest?definitionId=3&branchName=master

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1477847.svg
.. _zenodo: https://doi.org/10.5281/zenodo.1477847

.. |github| image:: https://img.shields.io/badge/GitHub-GPL--3.0-informational
.. _github: https://github.com/LiberTEM/LiberTEM/

.. |codeclimate| image:: https://api.codeclimate.com/v1/badges/dee042f64380f64737e5/maintainability
.. _codeclimate: https://codeclimate.com/github/LiberTEM/LiberTEM

.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.02006/status.svg
.. _joss: https://doi.org/10.21105/joss.02006

.. |precommit| image:: https://results.pre-commit.ci/badge/github/LiberTEM/LiberTEM/master.svg
.. _precommit: https://results.pre-commit.ci/latest/github/LiberTEM/LiberTEM/master

LiberTEM is an open source platform for high-throughput distributed processing
of large-scale binary data sets and live data streams using a modified
`MapReduce programming model <https://en.wikipedia.org/wiki/MapReduce>`_. The
current focus is `pixelated
<https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy#Universal_detectors_(4D_STEM)>`_
scanning transmission electron microscopy (`STEM
<https://en.wikipedia.org/wiki/Scanning_transmission_electron_microscopy>`_)
:cite:`doi:10.1002/9783527808465.EMC2016.6284,Ophus_2019` and scanning electron
beam diffraction data.

MapReduce-like processing allows to specify an algorithm through two functions:
One function that is mapped on portions of the input data, and another function
that merges (reduces) a partial result from this mapping step into the complete
result. A wide range of TEM and 4D STEM processing tasks can be expressed in
this fashion, see `Applications`_.

The UDF interface of LiberTEM offers a standardized, versatile API to decouple
the mathematical core of an algorithm from details of data source, parallelism,
and use of results. Mapping and merging can be performed in any order and with
different subdivisions of the input data, including running parts of the
calculation concurrently. That means the same implementation can be used in a
wide range of modalities, including massive scaling on clusters. Since each
merge step produces an intermediate result, this style of processing is suitable
for displaying live results from a running calculation in a GUI application and
for `processing live data streams <https://github.com/LiberTEM/LiberTEM-live>`_.
A closed-loop feedback between processing and instrument control can be realized
as well. See `User-defined functions
<https://libertem.github.io/LiberTEM/udf.html>`_ for more details on the
LiberTEM UDF interface.

The LiberTEM back-end offers `high throughput and scalability
<https://libertem.github.io/LiberTEM/architecture.html>`_ on PCs, single server
nodes, clusters and cloud services. On clusters it can use fast distributed
local storage on high-performance SSDs. That way it achieves `very high
aggregate IO performance
<https://libertem.github.io/LiberTEM/performance.html>`_ on a compact and
cost-efficient system built from stock components. All CPU cores and CUDA
devices in a system can be used in parallel.

LiberTEM is supported on Linux, Mac OS X and Windows. Other platforms that allow
installation of Python 3.6+ and the required packages will likely work as well. The
GUI is running in a web browser.

Installation
------------

The short version:

.. code-block:: shell

    $ virtualenv -p python3 ~/libertem-venv/
    $ source ~/libertem-venv/bin/activate
    (libertem-venv) $ python -m pip install "libertem[torch]"

    # optional for GPU support
    # See also https://docs.cupy.dev/en/stable/install.html
    (libertem-venv) $ python -m pip install cupy

Please see `our documentation
<https://libertem.github.io/LiberTEM/install.html>`_ for details!

Alternatively, to run the `LiberTEM Docker image
<https://libertem.github.io/LiberTEM/deployment/clustercontainer.html>`_:

.. code-block:: shell

    $ docker run -p localhost:9000:9000 --mount type=bind,source=/path/to/your/data/,dst=/data/,ro libertem/libertem

or

.. code-block:: shell

    $ singularity run docker://libertem/libertem -- /venv/bin/libertem-server

Deployment for offline data processing on a single-node system for a local user
is thoroughly tested and can be considered stable. Deployment on a cluster is
experimental and still requires some additional work, see `Issue #105
<https://github.com/LiberTEM/LiberTEM/issues/105>`_. Back-end support for live data processing
is still experimental as well, see https://github.com/LiberTEM/LiberTEM-live.

Applications
------------

Since LiberTEM is programmable through `user-defined functions (UDFs)
<https://libertem.github.io/LiberTEM/udf.html>`_, it can be used for a wide
range of processing tasks on array-like data and data streams. The following
applications have been implemented already:

- Virtual detectors (virtual bright field, virtual HAADF, center of mass :cite:`Krajnak2016`,
  custom shapes via masks)
- `Analysis of amorphous materials <https://libertem.github.io/LiberTEM/app/amorphous.html>`_
- `Strain mapping <https://libertem.github.io/LiberTEM-blobfinder/>`_
- `Off-axis electron holography reconstruction <https://libertem.github.io/LiberTEM/app/holography.html>`_
- `Single Side Band ptychography <https://ptychography-4-0.github.io/ptychography/>`_

Some of these applications are available through an `interactive web GUI
<https://libertem.github.io/LiberTEM/usage.html#gui-usage>`_. Please see `the
applications section <https://libertem.github.io/LiberTEM/applications.html>`_
of our documentation for details!

The Python API and user-defined functions (UDFs) can be used for complex
operations such as arbitrary linear operations and other features like data
export. Example Jupyter notebooks are available in the `examples directory
<https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_. If you are having
trouble running the examples, please let us know by filing an issue or
by `joining our Gitter chat <https://gitter.im/LiberTEM/Lobby>`_.

LiberTEM is suitable as a high-performance processing backend for other
applications, including live data streams. `Contact us
<https://gitter.im/LiberTEM/Lobby>`_ if you are interested!

LiberTEM is evolving rapidly and prioritizes features following user demand and
contributions. Currently we are working on `live data processing
<https://github.com/LiberTEM/LiberTEM-live>`_, `integration with Dask arrays and
Hyperspy <https://github.com/LiberTEM/LiberTEM/issues/922>`_, support for sparse
data, and implementing analysis methods for various applications of pixelated
STEM and other large-scale detector data. If you like to influence the direction
this project is taking, or if you'd like to `contribute
<https://libertem.github.io/LiberTEM/contributing.html>`_, please join our
`gitter chat <https://gitter.im/LiberTEM/Lobby>`_ and our `general mailing list
<https://groups.google.com/forum/#!forum/libertem>`_.

File formats
------------

LiberTEM currently opens most file formats used for pixelated STEM. See `our
general information on loading data
<https://libertem.github.io/LiberTEM/formats.html>`_ and `format-specific
documentation
<https://libertem.github.io/LiberTEM/reference/dataset.html#formats>`_ for more
information!

- Raw binary files
- Thermo Fisher EMPAD detector :cite:`Tate2016` files
- `Quantum Detectors MIB format <http://quantumdetectors.com/wp-content/uploads/2017/01/1532-Merlin-for-EM-Technical-Datasheet-v2.pdf>`_
- Nanomegas .blo block files
- Direct Electron DE5 files (HDF5-based) and Norpix SEQ files for `DE-Series <http://www.directelectron.com/de-series/>`_ detectors
- `Gatan K2 IS <https://web.archive.org/web/20180809021832/http://www.gatan.com/products/tem-imaging-spectroscopy/k2-camera>`_ raw format
- Stacks of Gatan DM3 and DM4 files (via `openNCEM <https://github.com/ercius/openNCEM>`_)
- FRMS6 from PNDetector pnCCD cameras :cite:`Simson2015` (currently alpha, gain correction still needs UI changes)
- FEI SER files (via `openNCEM <https://github.com/ercius/openNCEM>`_)
- MRC (via `openNCEM <https://github.com/ercius/openNCEM>`_)
- HDF5-based formats such as Hyperspy files, NeXus and EMD
- TVIPS binary files
- Please contact us if you are interested in support for an additional format!

Detectors (experimental)
------------------------

Currently the Quantum Detectors Merlin camera is supported for live processing.
Support for the Gatan K2 IS camera is in a prototype state. Please
`contact us <https://gitter.im/LiberTEM/Lobby>`_ if you are interested in this
feature! See https://github.com/LiberTEM/LiberTEM-live for more details on live
processing.

License
-------

LiberTEM is licensed under GPLv3. The I/O parts are also available under the MIT
license, please see LICENSE files in the subdirectories for details.
