Contributing
============

TODO

Running the Tests
-----------------

Our tests are written using pytest. For running them in a repeatable manner, we are using tox.
Tox automatically manages virtualenvs and allows testing on different Python versions and interpreter
implementations.

This makes sure that you can run the tests locally the same way as they are run in continuous integration.

After `installing tox <https://tox.readthedocs.io/en/latest/install.html>`_, you can run the tests on
all Python versions by simply running tox:

.. code-block:: shell

    $ tox

Or specify a specific environment you want to run:

.. code-block:: shell

    $ tox -e py36

On Windows
~~~~~~~~~~

On Windows with Anaconda, you have to create named aliases for the Python interpreter before you can run :literal:`tox` so that tox finds the python interpreter where it is expected. Assuming that you run LiberTEM with Python 3.6, place the following file as :literal:`python3.6.bat` in your LiberTEM conda environment base folder, typically :literal:`%LOCALAPPDATA%\\conda\\conda\\envs\\libertem\\`, where the :literal:`python.exe` of that environment is located.

.. code-block:: bat

    @echo off
    REM @echo off is vital so that the file doesn't clutter the output
    REM execute python.exe with the same command line
    @python.exe %*
    
To execute tests with Python 3.7, you create a new environment with Python 3.7:

.. code-block:: shell

    $ conda create -n libertem-3.7 python=3.7
    
Now you can create :literal:`python3.7.bat` in your normal LiberTEM environment alongside :literal:`python3.6.bat` and make it execute the Python interpreter of your new libertem-3.7 environment:

.. code-block:: bat

    @echo off
    REM @echo off is vital so that the file doesn't clutter the output
    REM execute python.exe in a different environment 
    REM with the same command line
    @%LOCALAPPDATA%\conda\conda\envs\libertem-3.7\python.exe %*

See also: http://tox.readthedocs.io/en/latest/developers.html#multiple-python-versions-on-windows

Code Style
----------

TODO

 * pep8


Building the Documentation
--------------------------

Documentation building is also done with tox, see above for the basics.
To start the live building process:

.. code-block:: shell

    $ tox -e docs

You can then view a live-built version at http://localhost:8008

