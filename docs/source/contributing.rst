Contributing
============

TODO

Updating Acknowledgments and Author List
----------------------------------------

Please update ``docs/source/acknowledgments.rst`` and ``.zenodo.json`` with your author information when you contribute to LiberTEM for the first time. This helps us to keep track of all contributors and give credit where credit is due! Please let us know if you wouldn't like to be credited.

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

Long-running tests such as starting a real local dask cluster are only executed if the environment variable `LT_RUN_FUNCTIONAL` is set. This variable is set for continuous integration on GitHub.

With bash:

.. code-block:: shell

    $ LT_RUN_FUNCTIONAL=1 tox

With Windows cmd:

.. code-block:: shell

    > set LT_RUN_FUNCTIONAL=1
    > tox

Running tests for the client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the testsuite for the client, first install the JavaScript/TypeScript dependencies:

.. code-block:: shell

   $ cd client/
   $ npm install

Then, in the same dircetory, to run the tests execute:

.. code-block:: shell

   $ npm test -- --coverage

This will run all tests and report code coverage. If you want to run the tests while developing the client,
you can run them in watch mode, which is the default:

.. code-block:: shell

   $ cd client/
   $ npm test

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

    > conda create -n libertem-3.7 python=3.7
    
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

Building the client
-------------------

The LiberTEM client is written in TypeScript, using a combination of React/Redux/Redux-Saga. The
client communicates with the Python API server using both HTTP and websockets. Because browsers
can't directly execute TypeScript, there is a build step involved, which translates the TypeScript
code into JavaScript that is then understood by the browser. 
This build step is needed both for development and then again for building the production version.

If you would like to contribute to the client, you first need to set up the development environment.
For this, first install nodejs. On Linux, we recommend to `install via package manager <https://nodejs.org/en/download/package-manager/>`_,
on Windows `the installer <https://nodejs.org/en/download/>`_ should be fine. Choose the current LTS version, which is 10.x at the time of writing.

One you have nodejs installed, you should have the npm command available in your path. You can then install
the needed build tools and dependencies by changing to the client directory and running the install command:

.. code-block:: shell

   $ cd client/
   $ npm install

.. note::
   
   It is always a good idea to start development with installing the current dependencies with the
   above command. Having old versions of dependencies installed may cause the build to fail or
   cause unpredictable failures.

Once this command finished without errors, you can start a development server:

.. code-block:: shell

   $ npm run start

This server watches all source files for changes and automatically starts the build process. This server,
which listens on port 3000, will only be able to serve requests for JavaScript and other static files -
for handling HTTP API requests you still need to run the Python libertem-server process.
Run it on the default port (9000) to allow proxying from the front-end server to the API server.

To learn more about the build process, please see `the README in the client directory <https://github.com/LiberTEM/LiberTEM/blob/master/client/README.md>`_.

You can then use any editor you like to change the client source files, in the client/src directory.
We recommend `visual studio code <https://code.visualstudio.com/>`_ for its excellent TypeScript support.

To simplify development and installing from a git checkout, we currently always ship a production build
of the client in the git repository. When you are creating a pull request for the client, please always
include a current production build. You can create it using a tox shortcut:

.. code-block:: shell

   $ tox -e build_client

This will build an optimized production version of the client and copy it into src/libertem/web/client.
This version will then be used when you start a libertem-server without the client development proxy in front.

Release checklist
-----------------

Not all aspects of LiberTEM are covered with automated unit tests. For that reason we should perform some manual tests before and after a release.

Before
~~~~~~

* Full documentation review and update
* `Confirm that wheel, tar.gz, and AppImage are built for the release candidate on GitHub <https://github.com/LiberTEM/LiberTEM/releases>`_
* Install release candidate packages from GitHub in a clean environment
* Correct version info displayed in info dialogue?
* Link check in version info dialogue
* Copy test files of all supported types to a fresh location or purge the parameter cache
    * Include floats, ints, big endian, little endian, complex raw data
* Open each test file
    * Are parameters recognized correctly, as far as implemented?
    * Any bad default values?
    * Does the file open correctly?
    * Have a look at the dataset info dialogue. Reasonable values?
* Perform all analyses on each test file.
    * Does the result change when the input parameters are changed?
    * All display channels present and looking reasonable?
    * Reasonable performance?
    * Use pick mode.
* Re-open all the files
    * Are the files listed in "recent files"?
    * Are the parameters filled from the cache correctly?
* Try opening all file types with wrong parameters
    * Proper understandable error messages?
* Pick one file and confirm keyboard and mouse interaction for all analyses
    * Correct bounds check for keyboard and mouse?
* Check what happens when trying to open non-existent files or directories in the GUI. 
    * Proper understandable error message?
    * Possible to continue working?
* Shut down libertem-server while analysis is running
    * Shut down within a few seconds?
    * All workers reaped?
* Check what happens when trying to open non-existent files by scripting.
    * Proper understandable error message? TODO automate?
* Check what happens when opening all file types with bad parameters by scripting
    * Proper understandable error message? TODO automate?
* Run all examples
* Check all examples in documentation, including API docstrings.
* Run libertem-server on Windows, connect to a remote dask cluster running on Linux, open all file types and perform an analysis for each file type.
* Use the GUI while a long-running analysis is running
    * Still usable, decent response times?

After releasing on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~

* Confirm that all release packages are built
* Install release package
* Confirm correct version info
* Upload to PyPi
* Upload to zenodo.org
* Update documentation with new links, if necessary
* Send announcement message on mailing list
