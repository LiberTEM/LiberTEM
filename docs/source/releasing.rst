Releasing
=========

This document describes release procedures and infrastructure that is relevant
for advanced contributors. See :ref:`contributing` for information on regular
contributions.

Tagging a version
-----------------

Install dependencies from :code:`scripts/requirements.txt`,
which are used by :code:`scripts/release`. Then call the script with
the :code:`bump` command, with the new version as parameter:

.. code-block:: shell

    $ ./scripts/release bump v0.3.0rc0 --tag

If you are bumping to a .dev0 suffix, omit :code:`--tag` and only pass :code:`--commit`:

.. code-block:: shell

    $ ./scripts/release bump v0.4.0.dev0 --commit

.. note::
   In normal development, the version in the master branch will be x.y.z.dev0,
   if the next expected version is x.y.z. When starting the release process, it
   will be bumped up to x.y.zrc0 (note: no dot before rc!) and possibly
   additional release candidates afterwards (rc1, ..., rcN). These release candidates
   are done mostly to assure our release scripts work as expected and for doing
   additional QA. See below for our QA process.

Issue template: release checklist
---------------------------------

When planning a release, create a new issue with the following checklist:

.. code-block:: text

    # Release checklist

    Issues and pull requests to be considered for this release:
    
    * #XXX
    * #YYY
    * #ZZZ

    ## Before (using a release candidate package)

    * [ ] Review open issues and pull requests
    * [ ] License review: no import of GPL code from MIT code
          `pydeps --only "libertem" --show-deps --noshow src\libertem | python scripts\licensecheck.py`
    * [ ] Run full CI pipeline, including slow tests, on [Azure DevOps](https://dev.azure.com/LiberTEM/LiberTEM/_build?definitionId=3)
    * [ ] Run tests for related packages w/ new LiberTEM version
        * [ ] LiberTEM-live
        * [ ] LiberTEM-holo
        * [ ] LiberTEM-blobfinder
        * [ ] ptychography40
    * [ ] Handle deprecation, search the code base for `DeprecationWarning`
          that are supposed to be removed in that release.
    * [ ] GUI dependency update with `npm install`
    * [ ] Review https://github.com/LiberTEM/LiberTEM/security/dependabot and update dependencies
    * [ ] Full documentation review and update, including link check using
          ``sphinx-build -b linkcheck "docs/source" "docs/build/html"``
    * [ ] Run complete test suite, including slow tests that are deactivated by default
          and tests that require sample files.
    * [ ] Update the expected version in notes on changes, i.e. from `0.3.0.dev0`
          to `0.3.0` when releasing version 0.3.0.
    * [ ] Update and review change log in `docs/source/changelog.rst`, merging
          snippets in `docs/source/changelog/*/` as appropriate.
    * [ ] Update the JSON files in the ``packaging/`` folder with author and project information
    * [ ] Edit `pytest.ini` to exclude flaky tests temporarily from release builds
    * [ ] Create a release candidate using `scripts/release`. See `scripts/release --help` for details.
    * [ ] `Confirm that wheel, tar.gz, and AppImage are built for the release candidate on
          GitHub <https://github.com/LiberTEM/LiberTEM/releases>`_
    * [ ] Confirm that a new version with the most recent release candidate is created in the
          `Zenodo.org sandbox <https://sandbox.zenodo.org/record/367108>`_ that is ready for submission.
    * [ ] Install release candidate packages in a clean environment
          (for example:
          `python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'libertem==0.2.0rc11'`)
    * [ ] Test the release candidate docker image
    * [ ] For the GUI-related items, open in an incognito window to start from a clean slate
    * [ ] Correct version info displayed in info dialogue?
    * [ ] Link check in version info dialogue
    * [ ] Make sure you have test files of all supported types available
        * [ ] Include floats, ints, big endian, little endian, complex raw data
    * [ ] Open each test file
        * [ ] Are parameters recognized correctly, as far as implemented?
        * [ ] Any bad default values?
        * [ ] Does the file open correctly?
        * [ ] Have a look at the dataset info dialogue. Reasonable values?
    * [ ] Perform all analyses on each test file.
        * [ ] Does the result change when the input parameters are changed?
        * [ ] All display channels present and looking reasonable?
        * [ ] Reasonable performance?
        * [ ] Use pick mode.
    * [ ] Re-open all the files
        * [ ] Are the files listed in "recent files"?
        * [ ] Are the parameters filled from the cache correctly?
    * [ ] Try opening all file types with wrong parameters
        * [ ] Proper understandable error messages?
    * [ ] Pick one file and confirm keyboard and mouse interaction for all analyses
        * [ ] Correct bounds check for keyboard and mouse?
    * [ ] Check what happens when trying to open non-existent files or directories in the GUI.
        * [ ] Proper understandable error message?
        * [ ] Possible to continue working?
    * [ ] Shut down libertem-server while analysis is running
        * [ ] Shut down within a few seconds?
        * [ ] All workers reaped?
    * [ ] Check what happens when trying to open non-existent files by scripting.
        * [ ] Proper understandable error message? TODO automate?
    * [ ] Check what happens when opening all file types with bad parameters by scripting
        * [ ] Proper understandable error message? TODO automate?
    * [ ] Run libertem-server on Windows, connect to a remote dask cluster running on Linux,
      open all file types and perform an analysis for each file type.
    * [ ] Use the GUI while a long-running analysis is running
        * [ ] Still usable, decent response times?
    * [ ] Confirm that pull requests and issues are handled as intended, i.e. milestoned and merged
      in appropriate branch.
    * [ ] Final version bump: `./scripts/release bump v0.3.0 --tag`, push to github
    * [ ] After pipeline finishes, write minimal release notes for the [release](https://github.com/liberTEM/LiberTEM/releases) and publish the GitHub release

    ## After releasing on GitHub

    * [ ] Confirm that all release packages are built and release notes are up-to-date
    * [ ] Install release package
    * [ ] Confirm correct version info
    * [ ] Confirm package upload to PyPI
    * [ ] Confirm images and tags on https://hub.docker.com/r/libertem/libertem
    * [ ] Publish new version on zenodo.org
    * [ ] Update documentation with new links, if necessary
        * [ ] Add zenodo badge for the new release to Changelog page
    * [ ] Send announcement message on mailing list
    * [ ] Edit `pytest.ini` to include flaky tests again
    * [ ] Bump version in master branch to next .dev0 (`./scripts/release bump v0.X.0.dev0 --commit`)
    * [ ] Add to institutional publication databases
    * [ ] Add the current LiberTEM version to [CVL](https://github.com/Chasdfracterisation-Virtual-Laboratory/CharacterisationVL-Software>) - add both the singularity and the .desktop file!
