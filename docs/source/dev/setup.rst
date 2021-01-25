Setting up a development environment
====================================

.. note::
    Distinguish between installing a released version and installing the latest
    development version. Both :ref:`installing from PyPi` and :ref:`installing from a git
    clone` use pip, but they do fundamentally different things. :code:`pip
    install libertem` downloads the latest release from PyPi.

    Changing directory to a git clone and running :code:`pip install -e .`
    installs from the local directory in editable mode. "Editable mode" means
    that the source directory is linked into the current Python environment
    rather than copied. That means changes in the source directory are
    immediately active in the Python environment.

    Installing from a git clone in editable mode is the correct setup for
    development work and using :ref:`the latest features in the development
    branch <continuous>`. Installing from PyPI is easier and preferred for
    production use and for new users.


When you want to develop LiberTEM itself, or run the latest git version, the installation works a
bit differently as opposed to installing from PyPi.
As a first step, please follow the instructions for creating a Python virtual environment from
the :ref:`installation` section. Then, instead of installing from PyPi, follow the instructions below.

.. _`installing from a git clone`:

Installing from a git clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to follow the latest development, you should install LiberTEM from
a git clone. As a prerequisite, you need to have git installed on your system. On Linux,
we suggest using the git package that comes with your package manager. On Windows, you can use one
of the many available clients, like  `git for windows <https://gitforwindows.org/>`_, 
`GitHub Desktop <https://desktop.github.com/>`_, `TortoiseGit <https://tortoisegit.org/>`_,
or the git integration of your development environment.

.. code-block:: shell

    $ git clone https://github.com/LiberTEM/LiberTEM

Or, if you wish to contribute to LiberTEM, create a fork of the LiberTEM repository
by following these steps:

#. Log into your `GitHub <https://github.com/>`_ account.

#. Go to the `LiberTEM GitHub <https://github.com/liberteM/LiberTEM/>`_ home page.

#. Click on the *fork* button:

    ..  figure:: ../images/forking_button.png

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

Installing extra dependencies works just like with PyPi:

.. code-block:: shell

    (libertem) $ pip install -e .[torch,hdbscan,cupy]

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
requirements for installed third-party packages have changed, you should re-run
``pip install -e .`` in order to install any missing packages.
