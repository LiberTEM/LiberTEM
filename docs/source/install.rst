.. _`installation`:

Installation
============

.. note::
    LiberTEM is currently working with Python 3.6 and Python 3.7. Support for
    Python 3.8 depends on an upcoming release of Dask.distributed. See also
    :issue:`452`, :pr:`482`.

.. note::
    Distinguish between installing a released version and installing the latest
    development version. Both `installing from PyPi`_ and `installing from a git
    clone`_ use pip, but they do fundamentally different things. :code:`pip
    install libertem` downloads the latest release from PyPi, which can be
    somewhat older.

    Changing directory to a git clone and running :code:`pip install -e .`
    installs from the local directory in editable mode. "Editable mode" means
    that the source directory is linked into the current Python environment
    rather than copied. That means changes in the source directory are
    immediately active in the Python environment.

    Installing from a git clone in editable mode is the correct setup for
    development work and using :ref:`the latest features in the development
    branch <continuous>`. Installing from PyPI is easier and preferred for new
    users.

.. include:: _single_node.rst

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
use virtualenvs or conda environments.

Using virtualenv
################

You can use `virtualenv <https://virtualenv.pypa.io/>`_ or `venv
<https://docs.python.org/3/tutorial/venv.html>`_ if you have a system-wide
Python 3.6 or 3.7 installation. For Mac OS X, using conda is recommended.

To create a new virtualenv for LiberTEM, you can use the following command:

.. code-block:: shell

    $ virtualenv -p python3.7 ~/libertem-venv/

Replace :code:`~/libertem-venv/` with any path where you would like to create
the venv. You can then activate the virtualenv with

.. code-block:: shell

    $ source ~/libertem-venv/bin/activate

Afterwards, your shell prompt should be prefixed with :code:`(libertem)` to
indicate that the environment is active:

.. code-block:: shell

    (libertem) $

For more information about virtualenv, for example if you are using a shell
without :code:`source`, please `refer to the virtualenv documentation
<https://virtualenv.pypa.io/en/stable/user_guide.html>`_. If you are often
working with virtualenvs, using a convenience wrapper like `virtualenvwrapper
<https://virtualenvwrapper.readthedocs.io/en/latest/>`_ is recommended.

Using conda
###########

If you are already using conda, or if you don't have a system-wide Python 3.6 or
3.7 installation, you can create a conda environment for LiberTEM.

This section assumes that you have `installed conda
<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
and that your installation is working.

You can create a new conda environment to install LiberTEM with the following
command:

.. code-block:: shell

    $ conda create -n libertem python=3.7

To install or later run LiberTEM, activate the environment with the following
command (see also :ref:`install on windows` if applicable):

.. code-block:: shell

    $ source activate libertem

Afterwards, your shell prompt should be prefixed with :code:`(libertem)` to
indicate that the environment is active:

.. code-block:: shell

    (libertem) $

Now the environment is ready to install LiberTEM.

For more information about conda, see their `documentation about creating and
managing environments
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Installing from PyPi
~~~~~~~~~~~~~~~~~~~~

To install the latest release version, you can use pip. Activate the Python
environment (conda or virtualenv) and install using:

.. code-block:: shell

    (libertem) $ pip install libertem

This should install LiberTEM and its dependencies in the environment. Please
continue by reading the :ref:`usage documentation`.

.. _`installing from a git clone`:

Installing from a git clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to follow the latest development, you should install LiberTEM from
a git clone:

.. code-block:: shell

    $ git clone https://github.com/LiberTEM/LiberTEM

Or if you wish to contribute to LiberTEM, follow these steps instead :

#. Log into your `GitHub <https://github.com/>`_ account.

#. Go to the `LiberTEM GitHub <https://github.com/liberteM/LiberTEM/>`_ home page.

#. Click on the *fork* button:

    ..  figure:: ./images/forking_button.png

#. Clone your fork of LiberTEM from GitHub to your computer

.. code-block:: shell

    $ git clone https://github.com/your-user-name/LiberTEM

For more information about `forking a repository
<https://help.github.com/en/github/getting-started-with-github/fork-a-repo>`_.
For a beginner-friendly introduction to git and GitHub, consider going through
the following resources:

* This `free course <https://www.udacity.com/course/version-control-with-git--ud123>`_
  covers the essentials of using Git.
* Practice `pull request <https://github.com/firstcontributions/first-contributions>`_
  in a safe sandbox environment.
* Sample `workflow <https://docs.astropy.org/en/latest/development/workflow/development_workflow.html>`_
  for contributing code.

Activate the Python environment (conda or virtualenv) and change to the newly
created directory with the clone of the LiberTEM repository. Now you can start
the LiberTEM installation. Please note the dot at the end, which indicates the
current directory!

.. code-block:: shell

    (libertem) $ pip install -e .

This should download the dependencies and install LiberTEM in the environment.
Please continue by reading the :ref:`usage documentation`.

PyTorch
~~~~~~~

LiberTEM can use `PyTorch <https://pytorch.org/>`_ for processing if it is
available. Otherwise it uses NumPy as a fallback. We've experienced up to 2x
speed-ups with PyTorch compared to a default NumPy installation. For that reason
we recommend `installing PyTorch <https://pytorch.org/>`_. We currently use
PyTorch only on the CPU. Contributions to use GPUs as well are very welcome!

You can let pip install PyTorch automatically by using the torch variant, for
example from PyPi:

.. code-block:: shell

    (libertem) $ pip install libertem[torch]

Or from git checkout:

.. code-block:: shell

    (libertem) $ pip install -e .[torch]

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

If you have installed from a git clone, you can easily update it to the current
status. Open a command line in the base directory of the LiberTEM clone and
update the source code with this command:

.. code-block:: shell

    $ git pull

The installation with ``pip install -e`` has installed LiberTEM in `"editable"
mode <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_.
That means the changes pulled from git are active immediately. Only if the
requirements for installed third-party packages have changed, you can re-run
``pip install -e .`` in order to install any missing packages.

After updating the installation, you can run the updated version by restarting
the libertem-server and afterwards reloading all browser windows that are
running the LiberTEM GUI.

.. _`install on windows`:

Windows
-------

The recommended method to install LiberTEM on Windows is based on `Miniconda 64
bit with Python version 3.6 or 3.7 <https://www.anaconda.com/distribution/>`_.
This installs a Python distribution.

For `installing from a git clone`_ you require a suitable git client, for
example `GitHub Desktop <https://desktop.github.com/>`_, `TortoiseGit
<https://tortoisegit.org/>`_, or `git for windows
<https://gitforwindows.org/>`_. Clone the repository
https://github.com/LiberTEM/LiberTEM in a folder of your choice.

From here on the installation and running of LiberTEM on Windows with the
Anaconda Prompt is very similar to `Using conda`_ on Linux or Mac OS X.

Differences:

* The command to activate a conda environment on Windows is

.. code-block:: shell

    > conda activate libertem

* You might have to install pip into your local LiberTEM conda environment to
  make sure that ``pip install`` installs packages into your local environment and
  not into the global Anaconda base environment. This helps to avoid permission
  issues and interference between environments.

.. code-block:: shell

    (libertem) > conda install pip

Jupyter
-------

To use the Python API from within a Jupyter notebook, you can install Jupyter
into your LiberTEM virtual environment.

.. code-block:: shell

    (libertem) $ pip install jupyter

You can then run a local notebook from within the LiberTEM environment, which
should open a browser window with Jupyter that uses your LiberTEM environment.

.. code-block:: shell

    (libertem) $ jupyter notebook

JupyterHub
----------

If you'd like to use the Python API from a LiberTEM virtual environment on a
system that manages logins with JupyterHub, you can easily `install a custom
kernel definition
<https://ipython.readthedocs.io/en/stable/install/kernel_install.html>`_ for
your LiberTEM environment.

First, you can launch a terminal on JupyterHub from the "New" drop-down menu in
the file browser. Alternatively you can execute shell commands by prefixing them
with "!" in a Python notebook.

In the terminal you can create and activate virtual environments and perform the
LiberTEM installation as described above. Within the activated LiberTEM
environment you additionally install ipykernel:

.. code-block:: shell

    (libertem) $ pip install ipykernel

Now you can create a custom ipython kernel definition for your environment:

.. code-block:: shell

    (libertem) $ python -m ipykernel install --user --name libertem --display-name "Python (libertem)"

After reloading the file browser window, a new Notebook option "Python
(libertem)" should be available in the "New" drop-down menu. You can test it by
creating a new notebook and running

.. code-block:: python

    In [1]: import libertem

Troubleshooting
---------------

If you are having trouble with the installation, please let us know by
either `filing an issue  <https://github.com/liberTEM/LiberTEM/issues>`_
or by asking on `our Gitter channel <https://gitter.im/LiberTEM/Lobby>`_.
