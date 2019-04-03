Contributing
============

TODO: introduction

Our code is hosted `on GitHub <https://github.com/libertem/libertem/>`_, and we are using 
`pull requests <https://help.github.com/en/articles/about-pull-requests>`_ to accept contributions.

Each pull request should focus on a single issue, to keep the number of changes small and reviewable.
To keep your changes organized and to prevent unrelated changes from disturbing your pull request,
create a new branch for each pull request. 

Before creating a pull request, please make sure all tests still pass. See `Running the Tests`_ for more
information. You should also update the test suite and add test cases for your contribution. See the section
`Code coverage`_ below on how to check if your new code is covered by tests.

To make sure our code base stays readable, we have follow a `Code Style`_.

Please update ``packaging/creators.json`` with your author information when you contribute to LiberTEM for the first time. This helps us to keep track of all contributors and give credit where credit is due! Please let us know if you wouldn't like to be credited. ``contributors.rst`` and  ``creators.rst`` in ``docs/source`` are generated from the JSON files with ``python scripts/build-authors-contributors``.

If you are changing parts of LiberTEM that are currently not covered by tests, please consider writing
new tests! When changing example code, which is not run as part of the tests, make sure the example
still runs.

When you have submitted your pull request, someone from the LiberTEM organization will review your
pull request, and may add comments or ask questions. If everything is good to go, your changes will
be merged and you can delete the branch you created for the pull request.

See also the `Guide on understanding the GitHub flow <https://guides.github.com/introduction/flow/>`_.



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

For faster iteration, you can also run only a part of the test suite, without using tox.
To make this work, first install the test requirements into your virtualenv:

.. code-block:: shell

   (libertem) $ pip install -r test_requirements.txt

Now you can run pytest on a subset of tests, for example:

.. code-block:: shell

   (libertem) $ pytest tests/test_analysis_masks.py

See the `pytest documentation <https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests>`_ for details on how to select which tests to run. Before submitting a pull request, you should always run the whole test suite.

Some tests are marked with `custom markers <https://docs.pytest.org/en/latest/example/markers.html>`_, for example we have some tests that take many seconds to complete.
To select tests to run by these marks, you can use the `-m` switch. For example, to only run the slow tests:

.. code-block:: shell

   $ tox -- -m slow

By default, these slow tests are not run. If you want to run both slow and all
other tests, you can use a boolean expression like this:

.. code-block:: shell

   $ tox -- -m "slow or not slow"

Another example, to exclude both slow and functional tests:

.. code-block:: shell

   $ tox -- -m "not functional and not slow"

In these examples, ``--`` separates the the arguments of tox (left of ``--``) from the arguments for pytest on the right.
List of marks used in our test suite:

- `slow`: tests that take much more than 1 second to run
- `functional`: tests that spin up a local dask cluster

Code coverage
-------------

After running the tests, you can inspect the test coverage by opening `htmlcov/index.html` in a web browser. When
creating a pull request, the change in coverage is also reported by the codecov bot. Ideally, the test coverage
should go up with each pull request, at least it should stay the same.

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

We try to keep our code `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ -compliant, with
line-length relaxed to 100 chars, and some rules ignored. See the flake8 section in setup.cfg
for the current PEP8 settings. As a general rule, try to keep your changes in a similar style
as the surrounding code.

You can check the code style by running:

.. code-block:: bat
   
   $ tox -e flake8


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
* Update the JSON files in the ``packaging/`` folder with author and project information
* Update ``contributors.rst`` and  ``creators.rst`` in ``docs/source`` from the JSON source files in ``packaging/`` using ``python scripts/build-authors-contributors``
* Update ``packaging/README.html`` with ``rst2html.py README.rst packaging/README.html`` and edit it in such a way that only the HTML body remains. This is used as a description on Zenodo.org
* `Confirm that wheel, tar.gz, and AppImage are built for the release candidate on GitHub <https://github.com/LiberTEM/LiberTEM/releases>`_
* Confirm that a new version is created on Zenodo.org that is ready for submission.
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
* Publish new version on zenodo.org
* Update documentation with new links, if necessary
* Send announcement message on mailing list
