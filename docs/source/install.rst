.. _`installation`:

Installation
============

.. note::
    LiberTEM can currently be used on Python >=3.9.3, 3.10, 3.11, 3.12, 3.13 and 3.14.

    If you would like to install the latest development version, please also
    see :ref:`installing from a git clone`.

LiberTEM is available to install through :code:`pip`:

.. code-block:: shell
    
    # Within a venv:
    (libertem-venv) $ pip install libertem


and :code:`conda` (via conda-forge):

.. code-block:: shell
    
    # Within a conda env:
    (libertem) $ conda install -c conda-forge libertem


Creating an isolated Python environment
---------------------------------------

It is good practice to use a dedicated virtual environment for LiberTEM
and its dependencies to avoid affecting other environments on your system.
To achieve this you can use a virtualenv or a conda environment, according
to preference.

Using virtualenv
^^^^^^^^^^^^^^^^

You can use `virtualenv <https://virtualenv.pypa.io/>`_ or `venv
<https://docs.python.org/3/tutorial/venv.html>`_ if you have a system-wide
compatible Python installation. For Mac OS X, using `conda`_ is recommended.

To create a new virtualenv for LiberTEM, you can use the following command:

.. code-block:: shell

    $ virtualenv -p python3 ~/libertem-venv/

If multiple Python versions are installed, replace :code:`python3` with 
:code:`python3.9` or a later version.

Replace :code:`~/libertem-venv/` with any path where you would like to create
the venv. You can then activate the virtualenv with

.. code-block:: shell

    $ source ~/libertem-venv/bin/activate

Afterwards, your shell prompt should be prefixed with :code:`(libertem-venv)` to
indicate that the environment is active:

.. code-block:: shell

    (libertem-venv) $

Now the environment is ready to install LiberTEM using
the :code:`pip` command at the top of this page.

For more information about virtualenv, for example if you are using a shell
without :code:`source`, please `refer to the virtualenv documentation
<https://virtualenv.pypa.io/en/stable/user_guide.html>`_. If you are often
working with virtualenvs, using a convenience wrapper like `virtualenvwrapper
<https://virtualenvwrapper.readthedocs.io/en/latest/>`_ is recommended.

.. _`conda`:

Using conda
^^^^^^^^^^^

If you are already using conda, or if you don't have a system-wide compatible
Python installation, you can create a conda environment for LiberTEM.

This section assumes that you have installed a conda-like environment manager, e.g.
`Miniforge <https://github.com/conda-forge/miniforge?tab=readme-ov-file#install>`_
and that your installation is working.

You can create a new environment to install LiberTEM with the following
command:

.. code-block:: shell

    $ conda create -n libertem python=3.12

Activate the environment with the following command:

.. code-block:: shell

    $ conda activate libertem

Afterwards, your shell prompt should be prefixed with :code:`(libertem)` to
indicate that the environment is active:

.. code-block:: shell

    (libertem) $

Now the environment is ready to install LiberTEM using
the :code:`conda` command at the top of this page.

For more information about conda, see their `documentation about creating and
managing environments
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Optional dependencies
---------------------

LiberTEM is capable of using additional dependencies to extend or enhance
its features.

.. _`cupy install`:

CuPy
^^^^

GPU support is based on `CuPy <https://cupy.dev/>`_. See the `CuPy installation
documentation <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_
for installation of a precompiled binary packages compatible with your GPU. This
is the recommended method to install CuPy, though it is also possible to
installs CuPy from source using:

.. code-block:: shell

    (libertem-venv) $ python -m pip install "libertem[cupy]"
    
though this requires a build chain and can be time-consuming.

PyTorch
^^^^^^^

LiberTEM can use `PyTorch <https://pytorch.org/>`_ for processing if it is
available, otherwise it uses NumPy as a fallback. We've experienced up to 2x
speed-ups with PyTorch compared to a default NumPy installation. For that reason
we recommend `installing PyTorch <https://pytorch.org/>`_. We currently use
PyTorch only on the CPU.

You can let pip install PyTorch automatically by using the torch variant, for
example from PyPI:

.. code-block:: shell

    (libertem-venv) $ python -m pip install "libertem[torch]"

.. versionadded:: 0.6.0

Other extra packages
--------------------

.. versionchanged:: 0.4.0
    A number of LiberTEM applications are being spun out as sub-packages that
    can be installed separately. See :ref:`packages` for an overview.

The full grid matching routines in :py:mod:`libertem.analysis.fullmatch` depend
on `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/>`_. This is an optional
dependency because of installation issues on some platforms.

Updating
--------

When installed from PyPI via pip, you can update like this:

.. code-block:: shell

    (libertem-venv) $ python -m pip install -U libertem

This should install a new version of LiberTEM and update all requirements that
have changed.

After updating the installation, you can run the updated version by restarting
the :code:`libertem-server` and afterwards reloading all browser windows that are
running the LiberTEM GUI. In other environments, like Jupyter notebooks, you
need to restart the Python interpreter to make sure the new version is used,
for example by restarting the ipython kernel.

.. _`airgapped`:

Air-gapped installation
-----------------------

Many microscope control computers are not connected to the internet, which means
that the usual installation methods don't work. It is not straightforward to
package a Python application into a self-contained executable or installer for
Windows, see also :issue:`39`. Furthermore, relocating Python environments, for
example with `conda-pack <https://conda.github.io/conda-pack/>`_ doesn't always
work reliably. In order to install LiberTEM on an air-gapped machine, you need a
computer with the same operating system, architecture and Python version as a
host system where you can prepare all required packages for the target system.

On the host
^^^^^^^^^^^

* Create a folder where you collect all required files, for example :code:`wheels`.
* If necessary, download and install a recent version of Python supported by
  LiberTEM and compatible with your systems from
  https://www.python.org/downloads/. This installer works without internet
  access.
* Start a new command shell and change into the folder for the required files.
* Confirm that you are using the intended Python version: :code:`python --version`
* Download and build all required packages for LiberTEM: :code:`python -m pip wheel libertem`
  You can add more packages and extras to this command as desired, for example Jupyter etc.
* Your folder :code:`wheels` should now contain all required Python packages.
* Transfer the folder :code:`wheels` and the Python installer to the target machine.

On the target
^^^^^^^^^^^^^

* If necessary, install the same Python version as on the host.
* Start a new command shell and confirm the Python version: :code:`python --version`
* Create a virtual environment in a folder of your choice: :code:`python -m venv libertem-venv`
* Activate the environment: On Windows cmd.exe :code:`libertem-venv\\Scripts\\activate.bat`
* Change to the directory with the Python packages.
* Install the LiberTEM package and other top-level packages
  using the package folder instead of the online package index:
  :code:`pip install --find-links . --no-index libertem`

Docker and Apptainer
--------------------

.. versionadded:: 0.9.0

A `Docker image with a LiberTEM installation
<https://ghcr.io/libertem/libertem>`_ is available on
the GitHub container registry. See :ref:`containers` for more details.

AppImage
--------

On Linux it is possible to install LiberTEM in a standalone form using the
provided AppImage file. To do this download the AppImage file from
`our releases page on GitHub <https://github.com/LiberTEM/LiberTEM/releases>`_,
mark it executable and run the AppImage. See also the `official documentation
<https://docs.appimage.org/user-guide/run-appimages.html>`_.

Troubleshooting
---------------

If you are having trouble with the installation, please let us know by
either `filing an issue  <https://github.com/liberTEM/LiberTEM/issues>`_
or by asking on `our Gitter channel <https://gitter.im/LiberTEM/Lobby>`_.

Integration and deployment
--------------------------

.. toctree::
    :maxdepth: 2

    deployment/jupyter
    deployment/clustercontainer
    deployment/as-a-service
