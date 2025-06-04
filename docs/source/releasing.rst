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

By default, the release script will refuse to downgrade the version. This can be
overridden with the :code:`--force` option, for example to prepare a point
release. In that case the repository should be prepared to contain only the
desired changes for the point release without any changes that are targeting the
next regular release, for example by cherry-picking into a stable branch.

.. code-block:: shell

    $ ./scripts/release bump v0.14.1rc0 --tag --force

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

    ## Step 0: before cutting the release candidate

    * [ ] Review open issues and pull requests
    * [ ] Confirm that pull requests and issues are handled as intended, i.e. milestoned and merged
          in appropriate branch.
    * [ ] License review: compatible with MIT license
    * [ ] Run full CI pipeline, including slow tests, on [Azure DevOps](https://dev.azure.com/LiberTEM/LiberTEM/_build?definitionId=3) and run the [Thorough workflow](https://github.com/LiberTEM/LiberTEM/actions/workflows/thorough.yml) on GitHub Actions
    * [ ] Handle deprecation, search the code base for `DeprecationWarning`
          that are supposed to be removed in that release.
    * [ ] GUI dependency update with `npm install`
    * [ ] Review https://github.com/LiberTEM/LiberTEM/security/dependabot and update dependencies
    * [ ] Full documentation review and update, including link check using
          ``sphinx-build -b linkcheck "docs/source" "docs/build/html"``
    * [ ] Update the expected version in notes on changes, i.e. from `0.3.0.dev0`
          to `0.3.0` when releasing version 0.3.0.
    * [ ] Update and review change log in `docs/source/changelog.rst`, merging
          snippets in `docs/source/changelog/*/` as appropriate.
    * [ ] Edit `pytest.ini` to exclude flaky tests temporarily from release
          builds, if there are currently any flaky tests
    * [ ] Update the JSON files in the ``packaging/`` folder with author and project information

    ## Step 1

    * [ ] Create a release candidate using `scripts/release`. See
          `scripts/release --help` for details. Example command:
          `./scripts/release bump v0.3.0rc1 --tag`, then push to GitHub
    * [ ] Confirm that wheel, tar.gz, and AppImage are built for the release candidate on
          [GitHub](https://github.com/LiberTEM/LiberTEM/releases)
    * [ ] Confirm that a new version with the most recent release candidate is created on
          [Zenodo.org](https://zenodo.org/doi/10.5281/zenodo.1477847) that is ready for submission.

    ## Step 2: using the release candidate package

    * [ ] Run tests for related packages w/ new LiberTEM version
        * [ ] LiberTEM-live
        * [ ] LiberTEM-holo
        * [ ] LiberTEM-blobfinder
        * [ ] ptychography40
        * [ ] LiberTEM-iCoM
    * [ ] Run complete test suite, including slow tests that are deactivated by default
          and tests that require sample files or CUDA support.
    * [ ] Install release candidate packages in a clean environment
          (for example:
          `python -m pip install 'libertem==0.2.0rc11'`)
    * [ ] Test the release candidate docker image
        * [ ] Confirm rc images and tags on https://ghcr.io/libertem/libertem
    * [ ] Quick GUI QA: open in an incognito window to start from a clean slate
        * [ ] Correct version info displayed in info dialogue?
        * [ ] Link check in version info dialogue
        * [ ] Test GUI without internet access
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
        * [ ] Run libertem-server on Windows, connect to a remote dask cluster running on Linux,
          open all file types and perform an analysis for each file type.
        * [ ] Use the GUI while a long-running analysis is running
            * [ ] Still usable, decent response times?
    * [ ] Check what happens when trying to open non-existent files by scripting.
        * [ ] Run `pytest -rA tests/io/datasets/test_missing.py` and check output
    * [ ] Check what happens when opening all file types with bad parameters by scripting
        * [ ] Run `pytest -rA tests/io/datasets/ -k "test_bad_params"` and check output

    ## Step 3: bump version and let release pipeline run

    * [ ] Final version bump: `./scripts/release bump v0.3.0 --tag`, push to github
    * [ ] After pipeline finishes, write minimal release notes for the [release](https://github.com/liberTEM/LiberTEM/releases) and publish the GitHub release

    ## Step 4: after releasing on GitHub

    * [ ] Confirm that all release packages are built and release notes are up-to-date
    * [ ] Install release package
    * [ ] Confirm correct version info
    * [ ] Confirm package upload to PyPI
    * [ ] Confirm images and tags on https://ghcr.io/libertem/libertem
    * [ ] Publish new version on zenodo.org
    * [ ] Update documentation with new links, if necessary
        * [ ] Add zenodo badge for the new release to Changelog page
    * [ ] Conda packaging: review PRs on https://github.com/conda-forge/libertem-feedstock/pulls
    * [ ] Send announcement message on mailing list
    * [ ] Edit `pytest.ini` to include flaky tests again
    * [ ] Bump version in master branch to next .dev0 (`./scripts/release bump v0.X.0.dev0 --commit`)
    * [ ] Add to institutional publication databases
