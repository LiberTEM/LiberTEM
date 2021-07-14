.. _`installation`:

Installation
============

.. note::
    LiberTEM can currently be used on Python 3.6, 3.7, 3.8 and >=3.9.3.

    If you would like to install the latest development version, please also
    see :ref:`installing from a git clone`.

The short version
-----------------

.. code-block:: shell

    $ virtualenv -p python3 ~/libertem-venv/
    $ source ~/libertem-venv/bin/activate
    (libertem-venv) $ python -m pip install "libertem[torch]"

    # optional for GPU support
    # See also https://docs.cupy.dev/en/stable/install.html
    (libertem-venv) $ python -m pip install cupy

For details, please read on!

Linux and Mac OS X
------------------

AppImage
~~~~~~~~

On Linux, the easiest method is to use the provided AppImage. Just download the
AppImage file from `our releases page on GitHub
<https://github.com/LiberTEM/LiberTEM/releases>`_, mark it executable and run
the AppImage. See also the `official documentation
<https://docs.appimage.org/user-guide/run-appimages.html>`_. Continue by reading
the :ref:`usage documentation`.

Creating an isolated Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To provide an isolated environment for LiberTEM and its dependencies, you can
use virtualenvs or conda environments. This is important if you want to use
different Python applications that may have conflicting dependencies, and it
allows to quickly re-create an environment in case things go sideways.

Using virtualenv
################

You can use `virtualenv <https://virtualenv.pypa.io/>`_ or `venv
<https://docs.python.org/3/tutorial/venv.html>`_ if you have a system-wide
compatible Python installation. For Mac OS X, using `conda`_ is recommended.

To create a new virtualenv for LiberTEM, you can use the following command:

.. code-block:: shell

    $ virtualenv -p python3 ~/libertem-venv/

If multiple Python versions are installed, replace :code:`python3` with 
:code:`python3.6` or a later version.

Replace :code:`~/libertem-venv/` with any path where you would like to create
the venv. You can then activate the virtualenv with

.. code-block:: shell

    $ source ~/libertem-venv/bin/activate

Afterwards, your shell prompt should be prefixed with :code:`(libertem-venv)` to
indicate that the environment is active:

.. code-block:: shell

    (libertem-venv) $

For more information about virtualenv, for example if you are using a shell
without :code:`source`, please `refer to the virtualenv documentation
<https://virtualenv.pypa.io/en/stable/user_guide.html>`_. If you are often
working with virtualenvs, using a convenience wrapper like `virtualenvwrapper
<https://virtualenvwrapper.readthedocs.io/en/latest/>`_ is recommended.

Continue by `installing from PyPI`_.

.. _`conda`:

Using conda
###########

If you are already using conda, or if you don't have a system-wide compatible
Python installation, you can create a conda environment for LiberTEM.

This section assumes that you have `installed anaconda or miniconda
<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
and that your installation is working.

You can create a new conda environment to install LiberTEM with the following
command:

.. code-block:: shell

    $ conda create -n libertem python=3.9

To install or later run LiberTEM, activate the environment with the following
command (see also :ref:`install on windows` if applicable):

.. code-block:: shell

    $ conda activate libertem

Afterwards, your shell prompt should be prefixed with :code:`(libertem)` to
indicate that the environment is active:

.. code-block:: shell

    (libertem) $

Now the environment is ready to install LiberTEM.

For more information about conda, see their `documentation about creating and
managing environments
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

.. _`installing from PyPI`:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

To install the latest release version, you can use pip. Activate the Python
environment (conda or virtualenv) and install using:

.. code-block:: shell

    (libertem) $ python -m pip install libertem

This should install LiberTEM and its dependencies in the environment. Please
continue by reading the :ref:`usage documentation`.

PyTorch
~~~~~~~

LiberTEM can use `PyTorch <https://pytorch.org/>`_ for processing if it is
available. Otherwise it uses NumPy as a fallback. We've experienced up to 2x
speed-ups with PyTorch compared to a default NumPy installation. For that reason
we recommend `installing PyTorch <https://pytorch.org/>`_. We currently use
PyTorch only on the CPU. Contributions to use GPUs as well are very welcome!

You can let pip install PyTorch automatically by using the torch variant, for
example from PyPI:

.. code-block:: shell

    (libertem) $ python -m pip install "libertem[torch]"

CuPy
~~~~

GPU support is based on `CuPy <https://cupy.dev/>`_. See
https://docs.cupy.dev/en/stable/install.html#installing-cupy for installation of
precompiled binary packages (recommended). :code:`python -m pip install
"libertem[cupy]"` installs CuPy from source, which requires a build chain and
can be time-consuming.

.. versionadded:: 0.6.0

Other extra packages
~~~~~~~~~~~~~~~~~~~~

.. versionchanged:: 0.4.0
    A number of LiberTEM applications are being spun out as sub-packages that
    can be installed separately. See :ref:`packages` for an overview.

The full grid matching routines in :py:mod:`libertem.analysis.fullmatch` depend
on `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/>`_. This is an optional
dependency because of installation issues on some platforms.

Updating
~~~~~~~~

When installed from PyPI via pip, you can update like this:

.. code-block:: shell

    (libertem) $ python -m pip install -U libertem

This should install a new version of LiberTEM and update all requirements that
have changed.

After updating the installation, you can run the updated version by restarting
the libertem-server and afterwards reloading all browser windows that are
running the LiberTEM GUI. In other environments, like jupyter notebooks, you
need to restart the Python interpreter to make sure the new version is used,
for example by restarting the ipython kernel.

.. _`install on windows`:

Windows
-------

The recommended method to install LiberTEM on Windows is based on `Miniconda 64
bit with a compatible Python version <https://www.anaconda.com/distribution/>`_.
This installs a Python distribution.

The installation and running of LiberTEM on Windows with the
Anaconda Prompt is very similar to `Using conda`_ on Linux or Mac OS X.

Differences:

* You might have to install pip into your local LiberTEM conda environment to
  make sure that ``pip install`` installs packages into your local environment and
  not into the global Anaconda base environment. This helps to avoid permission
  issues and interference between environments.

.. code-block:: shell

    (libertem) > conda install pip


Troubleshooting
---------------

If you are having trouble with the installation, please let us know by
either `filing an issue  <https://github.com/liberTEM/LiberTEM/issues>`_
or by asking on `our Gitter channel <https://gitter.im/LiberTEM/Lobby>`_.
