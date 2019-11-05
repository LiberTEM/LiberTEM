Releasing
=========

This document describes release procedures and infrastructure that is relevant
for advanced contributors. See :ref:`contributing` for information on regular
contributions.

Release checklist
-----------------

Not all aspects of LiberTEM are covered with automated unit tests. For that
reason we should perform some manual tests before and after a release.

Before (using a release candidate package)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Review open issues and pull requests
* Full documentation review and update, including link check using
  ``sphinx-build -b linkcheck "docs/source" "docs/build/html"``
* Update the expected version in notes on changes, i.e. from :code:`0.3.0.dev0`
  to :code:`0.3` when releasing version 0.3.
* Update and review change log in :code:`docs/source/changelog.rst`, merging
  snippets in :code:`docs/source/changelog/*/` as appropriate.
* Update the JSON files in the ``packaging/`` folder with author and project information
* Update ``contributors.rst`` and  ``creators.rst`` in ``docs/source`` from the JSON source
  files in ``packaging/`` using ``python scripts/build-authors-contributors``
* Create a release candidate using :code:`scripts/release`. See :code:`scripts/release --help` for details.
* `Confirm that wheel, tar.gz, and AppImage are built for the release candidate on
  GitHub <https://github.com/LiberTEM/LiberTEM/releases>`_
* Confirm that a new version with the most recent release candidate is created in the
  `Zenodo.org sandbox <https://sandbox.zenodo.org/record/367108>`_ that is ready for submission.
* Install release candidate packages in a clean environment
  (for example: 
  :code:`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'libertem==0.2.0rc11'`)
* For the GUI-related items, open in an incognito window to start from a clean slate
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
* Run libertem-server on Windows, connect to a remote dask cluster running on Linux,
  open all file types and perform an analysis for each file type.
* Use the GUI while a long-running analysis is running
    * Still usable, decent response times?
* Confirm that pull requests and issues are handled as intended, i.e. milestoned and merged
  in appropriate branch.

After releasing on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~

* Confirm that all release packages are built and release notes are up-to-date
* Install release package
* Confirm correct version info
* confirm package upload to PyPi
* Publish new version on zenodo.org
* Update documentation with new links, if necessary
* Send announcement message on mailing list
