Installation
============

.. include:: _single_node.rst

Linux and Mac OS X
------------------

Creating an isolated Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To provide an isolated environment for LiberTEM and its dependencies, you can use virtualenvs
or conda environments.

Using virtualenv
################

You can use virtualenv if you have a system wide Python 3.6 installation.
To create a new virtualenv for LiberTEM, you can use the following command:

.. code-block:: shell

    $ virtualenv -p python3.6 ~/libertem-venv/

Replace `~/libertem-venv/` with any path where you would like to create the venv. You can then activate
the virtualenv with

.. code-block:: shell
    
    $ source ~/libertem-venv/bin/activate

Afterwards, your shell prompt should be prefixed with `(libertem)` to indicate that the environment is active:

.. code-block:: shell

    (libertem) $ 

For more information about virtualenv, for example if you are using a shell without `source`, please
`refer to the virtualenv documentation <https://virtualenv.pypa.io/en/stable/userguide/#user-guide>`_.
If you are often working with virtualenvs, using a convenience wrapper like
`virtualenvwrapper <http://virtualenvwrapper.readthedocs.io/en/latest/>`_ is recommended.

Using conda
###########

If you are already using conda, or if you don't have a system-wide Python 3.6 installation,
you can create a conda environment for LiberTEM.

This section assumes that you have `installed conda <https://conda.io/docs/user-guide/install/index.html#regular-installation>`_
and that your installation is working.

You can create a new conda environment and activate it with the following commands:

.. code-block:: shell

    $ conda create -n libertem python=3.6
    $ source activate libertem

Afterwards, your shell prompt should be prefixed with `(libertem)` to indicate that the environment is active:

.. code-block:: shell

    (libertem) $ 

For more information about conda, see their `documentation about 
creating and managing environments <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.

Installing from a git clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to follow the latest development or contribute to LiberTEM, you should install LiberTEM
from a git clone:

.. code-block:: shell

    $ git clone https://github.com/LiberTEM/LiberTEM.git

In the newly created directory, with the python environment (conda or
virtualenv) activated, you can then start the LiberTEM installation:

.. code-block:: shell
    
    (libertem) $ pip install -e .

This should download the dependencies and install LiberTEM in the environment. Please
continue by reading the :doc:`Usage documentation <usage>`. 

Installing from PyPi
~~~~~~~~~

TODO: write once we have our package published in pypi

Windows
-------

TODO: adapt conda documentation from above


Troubleshooting
---------------

If you are having trouble with the installation, please let us know by
either `filing an issue  <https://github.com/liberTEM/LiberTEM/issues>`_
or by asking on `our Gitter channel <https://gitter.im/LiberTEM/Lobby>`_.
