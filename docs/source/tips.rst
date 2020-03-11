Tips and tricks
===============

.. _`ssh forwarding`:

Using SSH forwarding
--------------------

To securely connect to a remote instance of LiberTEM, you can use SSH
forwarding. For example with conda:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 <remote-hostname> "source activate libertem; libertem-server"

Or, with virtualenv:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 <remote-hostname> "/path/to/virtualenv/bin/libertem-server"

This makes LiberTEM, which is running on `remote-hostname`, available on your
local host via http://localhost:9000/


Running LiberTEM from an embedded interpreter
---------------------------------------------

If LiberTEM is run from within an embedded interpreter, the following steps
should be taken. This is necessary for Python scripting in Digital Micrograph,
for example.

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
