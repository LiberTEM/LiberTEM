Debugging
=========

.. testsetup:: *

    import numpy as np
    from libertem import api
    from libertem.executor.inline import InlineJobExecutor

    ctx = api.Context(executor=InlineJobExecutor())
    data = np.random.random((16, 16, 32, 32)).astype(np.float32)
    dataset = ctx.load("memory", data=data, sig_dims=2)
    roi = np.random.choice([True, False], dataset.shape.nav)

There are different parts of LiberTEM which can be debugged with different tools and methods.

Debugging the Web GUI
---------------------

For debugging the GUI, you can use all standard debugging tools for web development. Most useful
in this context are the `Chrome DevTools <https://developer.chrome.com/docs/devtools/>`_
or `Firefox Developer Tools <https://developer.mozilla.org/en-US/docs/Tools>`_, which can be
accessed by pressing F12. You can extend these with additional panels
`for React <https://reactjs.org/blog/2019/08/15/new-react-devtools.html>`_
and `for Redux <https://github.com/reduxjs/redux-devtools>`_.

These tools can be used for inspecting all frontend-related processes, from network traffic
up to rendering behavior. Take note of the :code:`/api/events/` websocket connection, where all
asynchronous notification and results will be transferred.

Note that you should always debug using the development build of the GUI, using :code:`npm start`,
as described in :ref:`the contributing section <building the client>`. Otherwise the debugging
experience may be painful, for example worse error output from react, minified source and
minified component names, ...

Debugging the API server
------------------------

If the API server returns a server error (500), the detailed exception should be logged
in the output of :code:`libertem-server`. You can also try
`enabling the debug mode of tornado <https://www.tornadoweb.org/en/stable/guide/running.html#debug-mode-and-automatic-reloading>`_
(there is currently no command line flag for this, so you need to change
:py:mod:`libertem.web.server` accordingly.)

If an analysis based on the exception alone is inconclusive,
you can try to reproduce the problem using the Python API and follow the instructions below.

.. _`debugging udfs`:

Debugging UDFs or other Python code
-----------------------------------

If you are trying to write a UDF, or debug other Python parts of LiberTEM, you can
instruct LiberTEM to use simple single-threaded execution using the
:class:`~libertem.executor.inline.InlineJobExecutor`.

.. testsetup::

    from libertem.udf.logsum import LogsumUDF

    udf = LogsumUDF()

.. testcode::

   from libertem.executor.inline import InlineJobExecutor
   from libertem import api as lt

   ctx = lt.Context(executor=InlineJobExecutor())

   ctx.run_udf(dataset=dataset, udf=udf)


You can then use all usual debugging facilities, including
`pdb <https://docs.python.org/3.7/library/pdb.html>`_ and
`the %pdb magic of ipython/Jupyter <https://ipython.org/ipython-doc/3/interactive/magics.html#magic-pdb>`_.

The :class:`libertem.executor.inline.InlineJobExecutor` uses a single CPU core
by default. It can be switched to GPU processing to test CuPy-enabled UDFs by
calling :meth:`libertem.common.backend.set_use_cuda` with the device ID to use.
:code:`libertem.common.backend.set_use_cpu(0)` switches back to CPU processing.

.. testsetup::

    from libertem.udf.masks import ApplyMasksUDF

    udf = ApplyMasksUDF(mask_factories=[lambda:np.ones(dataset.shape.sig)])

.. testcode::

   from libertem.executor.inline import InlineJobExecutor
   from libertem import api as lt
   from libertem.utils.devices import detect
   from libertem.common.backend import set_use_cpu, set_use_cuda

   ctx = lt.Context(executor=InlineJobExecutor())

   d = detect()
   if d['cudas'] and d['has_cupy']:
       set_use_cuda(d['cudas'][0])
   ctx.run_udf(dataset=dataset, udf=udf)
   set_use_cpu(0)

If a problem is only reproducible using the default executor, you will have to follow the
`debugging instructions of dask-distributed <https://docs.dask.org/en/latest/debugging.html>`_.
As the API server can't use the synchronous :class:`~libertem.executor.inline.InlineJobExecutor`,
this is also the case when debugging problems that only occur in context of the API server.

Debugging failing test cases
----------------------------

When a test case fails, there are some options to find the root cause:

The :code:`--pdb` command line switch of pytest can be used to automatically
drop you into a PDB prompt in the failing test case, where you will either land
on the failing :code:`assert` statement, or the place in the code where an
exception was raised.

This does not help if the test case only fails in CI. Here, it may be easier to
use logging. Because we call pytest with the :code:`--log-level=DEBUG`
parameter, the failing test case output will have a section containing the
captured logging output.

You can sprinkle the code with `log.debug(...)` calls that output the relevant
variables. In some cases you may also leave the logging statements in the code
even after the problem is fixed, depending on the overhead.
