.. _contributing:

Contributing
============

LiberTEM is intended and designed as a collaboratively developed platform for
data analysis. That means all our development is coordinated openly, mostly on
our `GitHub repository <https://github.com/LiberTEM/LiberTEM/>`_ where our code
is hosted. Any suggestions, Issues, bug reports, discussions and code
contributions are highly appreciated! Please let us know if you think we can
improve on something, be it code, communication or other aspects.

Development principles
----------------------

We have a `rather extensive and growing list of things to work on
<https://github.com/LiberTEM/LiberTEM/issues>`_ and therefore have to prioritize
our limited resources to work on items with the largest benefit for our user
base and project. Supporting users who contribute code is most important to us.
Please contact us for help! Furthermore, we prioritize features that create
direct benefits for many current users or open significant new applications.
Generally, we follow user demand with our developments.

For design of new features we roughly follow the `lead user method
<https://en.wikipedia.org/wiki/Lead_user>`_, which means that we develop new
features closely along a non-trivial real-world application in order to make
sure the developments are appropriate and easy to use in practice. The interface
for :ref:`user-defined functions`, as an example, follows the requirements
around implementing and running complex algorithms like `strain mapping
<https://libertem.github.io/LiberTEM-blobfinder/examples.html>`_ for distributed
systems.

Furthermore we value a systematic approach to development with requirements
analysis and evaluation of design options as well as iterative design with fast
test and review cycles.

Code contributions
------------------

We are using `pull requests
<https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
to accept contributions. Each pull request should focus on a single issue, to
keep the number of changes small and reviewable. To keep your changes organized
and to prevent unrelated changes from disturbing your pull request, create a new
branch for each pull request.

All pull requests should come from a user's personal fork since we don't push to
the main repository for development. See the `GitHub documentation on how to
create and manage forks
<https://docs.github.com/en/github/getting-started-with-github/quickstart/fork-a-repo>`_
for details.

Before creating a pull request, please make sure all tests still pass. See
`Running the Tests`_ for more information. You should also update the test suite
and add test cases for your contribution. See the section `Code coverage`_ below
on how to check if your new code is covered by tests.

To make sure our code base stays readable and consistent, we follow a `Code Style`_.

Please update ``packaging/creators.json`` with your author information when you
contribute to LiberTEM for the first time. This helps us to keep track of all
contributors and give credit where credit is due! Please let us know if you
wouldn't like to be credited. Our :ref:`authorship` describes in more detail how
we manage authorship of LiberTEM and related material.

If you are changing parts of LiberTEM that are currently not covered by tests,
please consider writing new tests! When changing example code, which is not run
as part of the tests, make sure the example still runs.

When adding or changing a feature, you should also update the corresponding
documentation, or add a new section for your feature. Follow the current
documentation structure, or ask the maintainers where your new documentation
should end up. When introducing a feature, it is okay to start with a draft
documentation in the first PR, if it will be completed later. Changes of APIs
should update the corresponding docstrings.

Please include version information if you add or change a feature in order to
track and document changes. We use a rolling documentation that documents
previous behavior as well, for example *This feature was added in Version
0.3.0.dev0* or *This describes the behavior from Version 0.3.0.dev0 and onwards.
The previous behavior was this and that*. If applicable, use
`versionadded <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded>`_
and related directives.

The changelog for the development branch is maintained as a collection of files
in the :code:`docs/source/changelog/*/` folder structure. Each change should get
a separate file to avoid merge conflicts. The files are merged into the
master changelog when creating a release.

The following items might require an
update upon introducing or changing a feature:

* Changelog snippet in :code:`docs/source/changelog/*/`
* Docstrings
* Examples
* Main Documentation

When you have submitted your pull request, someone from the LiberTEM
organization will review your pull request, and may add comments or ask
questions. 

In case your PR touches I/O code, an organization member may run
the I/O tests with access to test data sets on a separate Azure Agent,
using the following comment on the PR:

.. code-block:: text

    /azp run libertem.libertem-data

If everything is good to go, your changes will be merged and you can
delete the branch you created for the pull request.

.. seealso:: `Guide on understanding the GitHub flow <https://guides.github.com/introduction/flow/>`_

.. seealso:: `How to make a simple GitHub PR (video) <https://www.youtube.com/watch?v=cysuuUtbC6E>`_

.. _`running tests`:

Running the tests
-----------------

Our tests are written using pytest. For running them in a repeatable manner, we
are using tox. Tox automatically manages virtualenvs and allows testing on
different Python versions and interpreter implementations.

This makes sure that you can run the tests locally the same way as they are run
in continuous integration.

After `installing tox <https://tox.readthedocs.io/en/latest/install.html>`_, you
can run the tests on all Python versions by simply running tox:

.. code-block:: shell

    $ tox

Or specify a specific environment you want to run:

.. code-block:: shell

    $ tox -e py36

For faster iteration, you can also run only a part of the test suite, without
using tox. To make this work, first install the test requirements into your
virtualenv:

.. code-block:: shell

   (libertem) $ python -m pip install -r test_requirements.txt

Now you can run pytest on a subset of tests, for example:

.. code-block:: shell

   (libertem) $ pytest tests/test_analysis_masks.py

Or you can run tests in parallel, which may make sense if you have a beefy
machine with many cores and a lot of RAM:

.. code-block:: shell

   (libertem) $ pytest -n auto tests/

See the `pytest documentation
<https://docs.pytest.org/en/latest/how-to/usage.html#specifying-which-tests-to-run>`_
for details on how to select which tests to run. Before submitting a pull
request, you should always run the whole test suite.

Some tests are marked with `custom markers
<https://docs.pytest.org/en/latest/example/markers.html>`_, for example we have
some tests that take many seconds to complete. To select tests to run by these
marks, you can use the `-m` switch. For example, to only run the slow tests:

.. code-block:: shell

   $ tox -- -m slow

By default, these slow tests are not run. If you want to run both slow and all
other tests, you can use a boolean expression like this:

.. code-block:: shell

   $ tox -- -m "slow or not slow"

Another example, to exclude both slow and functional tests:

.. code-block:: shell

   $ tox -- -m "not functional and not slow"

In these examples, ``--`` separates the the arguments of tox (left of ``--``)
from the arguments for pytest on the right. List of marks used in our test
suite:

- `slow`: tests that take much longer than 1 second to run
- `functional`: tests that spin up a local dask cluster

Example notebooks
~~~~~~~~~~~~~~~~~

The example notebooks are also run as part of our test suite using `nbval
<https://nbval.readthedocs.io/en/latest/>`_. The
output saved in the notebooks is compared to the output of re-running the
notebook. When writing an example notebook, this sometimes requires some work,
because certain things will change from run to run. Please check `the nbval
documentation
<https://nbviewer.jupyter.org/github/computationalmodelling/nbval/blob/master/docs/source/index.ipynb>`_
to understand how to ignore such values. See also the :code:`nbval_sanitize.cfg`
file for our currently ignored patterns.

If your notebook outputs floating point values, you may get spurious failures
from precision issues. You can set the precision using the :code:`%precision` ipython
magic, which should be used *after* importing numpy.

You can run the notebook tests as follows:

.. code-block:: shell

    $ TESTDATA_BASE_PATH=/path/to/the/test/data tox -e notebooks

You will need access to certain sample data sets; as most of them are not
published yet, please contact us to get access!

CUDA
~~~~

To run tests that require CuPy using tox, you can specify the CUDA version with the test environment:

.. code-block:: shell

    $ tox -e py36-cuda101

Code coverage
~~~~~~~~~~~~~

After running the tests, you can inspect the test coverage by opening
`htmlcov/index.html` in a web browser. When creating a pull request, the change
in coverage is also reported by the codecov bot. Ideally, the test coverage
should go up with each pull request, at least it should stay the same.

.. _`benchmarking`:

Benchmarking
~~~~~~~~~~~~

LiberTEM uses `pytest-benchmark
<https://pytest-benchmark.readthedocs.io/en/latest/usage.html>`_ to benchmark
certain performance-critical parts of the code. You can run the benchmarks
ad-hoc using

.. code-block:: shell

   $ pytest benchmarks/

The benchmarks for Numba compilation time are disabled by default since Numba
caches compilation results, i.e. one has to make sure that benchmarked functions
were not previously run in the same interpreter. To run them, you can use

.. code-block:: shell

   $ pytest -m compilation benchmarks/

In order to record a complete benchmark run for later comparison, you can use

.. note::
   This requires :code:`tox>=3.15` since we are using generative section names
   in :code:`tox.ini`.

.. code-block:: shell

   $ tox -e benchmark
   $ # alternatively
   $ tox -e benchmark-cuda101
   $ tox -e benchmark-cuda102

This saves the benchmark data as a JSON file in a subfolder of
:code:`benchmark_results`. A process to commit such results and report them in a
convenient fashion is to be developed. See :issue:`198`, feedback welcome!

.. versionadded:: 0.6.0
   First benchmarks included to help resolve :issue:`814`, benchmark coverage will grow over time.

Running tests for the client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the testsuite for the client, first install the JavaScript/TypeScript dependencies:

.. code-block:: shell

   $ cd client/
   $ npm install

Then, in the same directory, to run the tests execute:

.. code-block:: shell

   $ npm test -- --coverage

This will run all tests and report code coverage. If you want to run the tests
while developing the client, you can run them in watch mode, which is the
default:

.. code-block:: shell

   $ cd client/
   $ npm test

Code style
----------

We try to keep our code `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_
-compliant, with line-length relaxed to 100 chars, and some rules ignored. See
the flake8 section in :code:`setup.cfg` for the current PEP8 settings. As a
general rule, try to keep your changes in a similar style as the surrounding
code.

Before submitting a pull request, please check the code style by running:

.. code-block:: shell

   $ pre-commit run

You may need to install `pre-commit <https://pre-commit.com/>`_ into your
Python environment first.
We recommend using an editor that can check code style on the fly, such as
`Visual Studio Code <https://code.visualstudio.com/docs/python/linting>`__.

Docstrings
~~~~~~~~~~

The `NumPy docstring guide
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ is
our guideline for formatting docstrings. We are testing docstring code examples
in Continuous Integration using `doctest
<https://docs.python.org/3/library/doctest.html>`_. You can test files by hand
by running :code:`pytest --doctest-modules <pathspec>`.

Building the documentation
--------------------------

Documentation building is also done with tox, see above for the basics. It
requires manual `installation of pandoc <https://pandoc.org/installing.html>`_
on the build system since pandoc can't be installed reliably using pip. To start
the live building process:

.. code-block:: shell

    $ tox -e docs

You can then view the documentation at http://localhost:8008, which will
be rebuilt every time a change to the documentation source code is detected.
Note that changes to the Python source code don't trigger a rebuild, so if
you are working on docstrings, you may have to manually trigger a rebuild,
for example by saving one of the :code:`.rst` files.

You can include code samples with the `doctest sphinx extension
<https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ and test
them with

.. code-block:: shell

    $ tox -e docs-check

.. _`building the client`:

Building the GUI (client)
-------------------------

The LiberTEM GUI (also called the client) is written in TypeScript, using a combination of
React/Redux/Redux-Saga. The client communicates with the Python API server using
both HTTP and websockets. Because browsers can't directly execute TypeScript,
there is a build step involved, which translates the TypeScript code into
JavaScript that is then understood by the browser. This build step is needed
both for development and then again for building the production version.

If you would like to contribute to the client, you first need to set up the
development environment. For this, first install Node.js. On Linux, we recommend
to `install via package manager
<https://nodejs.org/en/download/package-manager/>`_, on Windows `the installer
<https://nodejs.org/en/download/>`_ should be fine. Choose the current LTS
version.

One you have Node.js installed, you should have the :code:`npm` command available
in your path. You can then install the needed build tools and dependencies by
changing to the client directory and running the install command:

.. code-block:: shell

   $ cd client/
   $ npm install

.. note::

   It is always a good idea to start development with installing the current
   dependencies with the above command. Having old versions of dependencies
   installed may cause the build to fail or cause unpredictable failures.

Once this command finished without errors, you can start a development server
(also from the client directory):

.. code-block:: shell

   $ npm run start

This server watches all source files for changes and automatically starts the
build process. The development server, which listens on port 3000, will only be
able to serve requests for JavaScript and other static files. For handling HTTP
API requests you still need to run the Python :code:`libertem-server` process on
the default port (9000) alongside the development server:

.. code-block:: shell

   $ libertem-server --no-browser

This allows proxying the HTTP API requests from the front-end server to the API
server without opening an additional browser window that could interfere with
the development server.

To learn more about the build process, please see `the README in the client
directory <https://github.com/LiberTEM/LiberTEM/blob/master/client/README.md>`_.

You can then use any editor you like to change the client source files, in the
:code:`client/src` directory. We recommend `Visual Studio Code
<https://code.visualstudio.com/>`_ for its excellent TypeScript support.

To simplify development and installing from a git checkout, we currently always
ship a production build of the client in the git repository. Please always open
your pull request for the client as WIP and include a rebuilt production build
after the PR is approved and ready to merge. You can create it using a tox
shortcut:

.. code-block:: shell

   $ tox -e build_client

This will build an optimized production version of the client and copy it into
:code:`src/libertem/web/client`. This version will then be used when you start a
libertem-server without the client development proxy in front.

Advanced
--------

See more:

.. toctree::
   :maxdepth: 2

   releasing
   dist_tests
