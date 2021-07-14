Tips and tricks
===============

This is a collection of various helpful tips that don't fit in elsewhere.

.. _`ssh forwarding`:

Using SSH forwarding
--------------------

As there is currently no built-in authentication yet, listening on a different host than
:code:`127.0.0.1` / :code:`localhost` is disabled. As a workaround, if you want
to access LiberTEM from a different computer, you can use ssh port forwarding.
For example with conda:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 <remote-hostname> "source activate libertem; libertem-server"

Or, with virtualenv:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 <remote-hostname> "/path/to/virtualenv/bin/libertem-server"

This makes LiberTEM, which is running on `remote-hostname`, available on your
local host via http://localhost:9000/


Activating iywidgets in Jupyter
-------------------------------

Some examples use :code:`ipywidgets` in notebooks, most notably the fast
:class:`~libertem.viz.bqp.BQLive2DPlot`. In some cases the corresponding Jupyter
Notebook extension `has to be activated manually
<https://ipywidgets.readthedocs.io/en/stable/user_install.html#installing-in-classic-jupyter-notebook>`_:

.. code-block:: shell

    $ jupyter nbextension enable --py widgetsnbextension

Running in a top-level script
-----------------------------

Since LiberTEM uses multiprocessing, the `script entry point may have to be
protected
<https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>`_,
most notably on Windows:

.. testcode::

    if __name__ == '__main__':
        # Here goes starting a LiberTEM Context etc
        ...

This is not necessary if LiberTEM is used in a Jupyter notebook or IPython.

Gatan Digital Micrograph and other embedded interpreters
--------------------------------------------------------

If LiberTEM is run from within an embedded interpreter, the following steps
should be taken. This is necessary for Python scripting in Gatan Digital
Micrograph (GMS), for example.

The variable :code:`sys.argv` `may not be set in embedded interpreters
<https://bugs.python.org/issue32573>`_, but it is expected by the
:code:`multiprocessing` module when spawning new processes. This workaround
guarantees that :code:`sys.argv` is set `until this is fixed upstream
<https://github.com/python/cpython/pull/12463>`_:

.. testsetup::

    import sys

.. testcode::

    if not hasattr(sys, 'argv'):
        sys.argv  = []

Furthermore, the `correct executable for spawning subprocesses
<https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_executable>`_
has to be set.

.. testsetup::

    import multiprocessing
    import sys
    import os

.. testcode::

    multiprocessing.set_executable(
        os.path.join(sys.exec_prefix, 'pythonw.exe'))  # Windows only

In GMS the script may have to run in an additional thread since loading SciPy in
a GMS background thread doesn't work. See https://www.gatan.com/python-faq for
more information.

.. testcode::

    import threading

    def main():
        # Here goes the actual script
        ...

    if __name__ == '__main__':
        # Start the workload "main()" in a thread and wait for it to finish
        th = threading.Thread(target=main)
        th.start()
        th.join()

See `our examples folder
<https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_ for a number of
scripts that work in GMS!

.. _`show warnings`:

Show deprecation warnings
-------------------------

Many warning messages via the :code:`warnings` built-in module are suppressed by
default, including in interactive shells such as IPython and Jupyter. If you'd
like to be informed early about upcoming backwards-incompatible changes, you
should activate deprecation warnings. This is recommended since LiberTEM is
under active development.

.. testcode::

    import warnings

    warnings.filterwarnings("default", category=DeprecationWarning)
    warnings.filterwarnings("default", category=PendingDeprecationWarning)

.. _`profiling tests`:

Profiling long-running tests
----------------------------

Since our code base and test coverage is growing continuously, we should make
sure that our test suite remains efficient to finish within reasonable time
frames.

You can find the five slowest tests in the output of Tox, see :ref:`running tests`
for details. If you are using :code:`pytest` directly, you can use the
:code:`--durations` parameter:

.. code-block:: text

    (libertem) $ pytest --durations=10 tests/
    (...)
    ================= slowest 10 test durations =============================
    31.61s call     tests/udf/test_blobfinder.py::test_run_refine_affinematch
    17.08s call     tests/udf/test_blobfinder.py::test_run_refine_sparse
    16.89s call     tests/test_analysis_masks.py::test_numerics_fail
    12.78s call     tests/server/test_job.py::test_run_job_delete_ds
    10.90s call     tests/server/test_cancel.py::test_cancel_udf_job
     8.61s call     tests/test_local_cluster.py::test_start_local
     8.26s call     tests/server/test_job.py::test_run_job_1_sum
     6.76s call     tests/server/test_job.py::test_run_with_all_zeros_roi
     6.50s call     tests/test_analysis_masks.py::test_numerics_succeed
     5.75s call     tests/test_analysis_masks.py::test_avoid_calculating_masks_on_client
    = 288 passed, 66 skipped, 6 deselected, 2 xfailed, 7 warnings in 260.65 seconds =

Please note that functional tests which involve starting a local cluster have
long lead times that are hard to avoid.

In order to gain more information on what slows down a particular test, you can
install the `pytest-profiling extension
<https://github.com/man-group/pytest-plugins/tree/master/pytest-profiling>`_ and
use it to profile individual slow tests that you identified before:

.. code-block:: text

    (libertem) $ pytest --profile tests/udf/test_blobfinder.py::test_run_refine_affinematch
    (...)
    749921 function calls (713493 primitive calls) in 5.346 seconds

    Ordered by: cumulative time
    List reduced from 1031 to 20 due to restriction <20>

    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         1    0.000    0.000    5.346    5.346 runner.py:76(pytest_runtest_protocol)
     44/11    0.000    0.000    5.344    0.486 hooks.py:270(__call__)
     44/11    0.000    0.000    5.344    0.486 manager.py:65(_hookexec)
     44/11    0.000    0.000    5.344    0.486 manager.py:59(<lambda>)
     44/11    0.001    0.000    5.344    0.486 callers.py:157(_multicall)
         1    0.000    0.000    5.331    5.331 runner.py:83(runtestprotocol)
         3    0.000    0.000    5.331    1.777 runner.py:172(call_and_report)
         3    0.000    0.000    5.330    1.777 runner.py:191(call_runtest_hook)
         3    0.000    0.000    5.329    1.776 runner.py:219(from_call)
         3    0.000    0.000    5.329    1.776 runner.py:198(<lambda>)
         1    0.000    0.000    5.138    5.138 runner.py:119(pytest_runtest_call)
         1    0.000    0.000    5.138    5.138 python.py:1355(runtest)
         1    0.000    0.000    5.138    5.138 python.py:155(pytest_pyfunc_call)
         1    0.004    0.004    5.137    5.137 test_blobfinder.py:149(test_run_refine_affinematch)
         5    0.159    0.032    3.150    0.630 generate.py:6(cbed_frame)
       245    0.001    0.000    2.989    0.012 masks.py:98(circular)
       245    0.046    0.000    2.988    0.012 masks.py:8(_make_circular_mask)
       245    0.490    0.002    2.941    0.012 masks.py:280(radial_bins)
       245    0.152    0.001    2.229    0.009 masks.py:212(polar_map)
        25    0.001    0.000    1.968    0.079 blobfinder.py:741(run_refine)

    =============================== 1 passed, 1 warnings in 7.81 seconds ============================

.. _`os mismatch`:

Platform-dependent code and remote executor
-------------------------------------------

Platform-dependent code in a lambda function or nested function can lead to
incompatibilities when run on an executor with remote workers, such as the
:class:`~libertem.executor.dask.DaskJobExecutor`. Instead, the function should
be defined as part of a module, for example as a stand-alone function or as a
method of a class. That way, the correct remote implementation for
platform-dependent code is used on the remote worker since only a reference to
the function and not the implementation itself is sent over.

Benchmark Numba compilation time
--------------------------------

One has to capture the very first execution of a jitted function and compare it
with subsequent executions to measure its compilation time. By default,
pytest-benchmark performs calibration runs and possibly warmup rounds that don't
report the very first run.

The only way to completely disable this is to use the `pedantic mode
<https://pytest-benchmark.readthedocs.io/en/latest/pedantic.html>`_ specifying
no warmup rounds and two rounds with one iteration each:

.. code-block:: python

   @numba.njit
    def hello():
        return "world"


    @pytest.mark.compilation
    @pytest.mark.benchmark(
        group="compilation"
    )
    def test_numba_compilation(benchmark):
        benchmark.extra_info["mark"] = "compilation"
        benchmark.pedantic(hello, warmup_rounds=0, rounds=2, iterations=1)

That way the maximum is the first run with compilation, and the minimum is the
second one without compilation. Tests are marked as compilation tests in the
extra info as well to aid later data evaluation. Note that the compilation tests
will have poor statistics since it only runs once. If you have an idea on how to
collect better statistics, please `let us know
<https://github.com/LiberTEM/LiberTEM/issues/new>`_!


Simulating slow systems with control groups
-------------------------------------------

Under Linux, it is possible to simulate a slow system using control groups:

.. code-block:: shell

    sudo cgcreate -g cpu:/slow
    sudo cgset -r cpu.cfs_period_us=1000000 slow
    sudo cgset -r cpu.cfs_quota_us=200000 slow
    sudo chown root:<yourgroup> /sys/fs/cgroup/cpu,cpuacct/slow
    sudo chmod 664 /sys/fs/cgroup/cpu,cpuacct/slow

Then, as a user, you can use :code:`cgexec` to run a command in that control group:

.. code-block:: shell

    cgexec -g cpu:slow pytest tests/

This is useful, for example, to debug test failures that only seem to happen in CI
or under heavy load. Note that tools like :code:`cgcreate` only work with cgroups v1,
with newer distributions using cgroups v2 you may have to adapt these instructions.

.. _`jupyter install`:

Jupyter
-------

To use the Python API from within a Jupyter notebook, you can install Jupyter
into your LiberTEM virtual environment.

.. code-block:: shell

    (libertem) $ python -m pip install jupyter

You can then run a local notebook from within the LiberTEM environment, which
should open a browser window with Jupyter that uses your LiberTEM environment.

.. code-block:: shell

    (libertem) $ jupyter notebook

.. _`jupyterhub install`:

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

    (libertem) $ python -m pip install ipykernel

Now you can create a custom ipython kernel definition for your environment:

.. code-block:: shell

    (libertem) $ python -m ipykernel install --user --name libertem --display-name "Python (libertem)"

After reloading the file browser window, a new Notebook option "Python
(libertem)" should be available in the "New" drop-down menu. You can test it by
creating a new notebook and running

.. code-block:: python

    In [1]: import libertem
