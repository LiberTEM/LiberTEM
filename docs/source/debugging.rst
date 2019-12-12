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
in this context are the `Chrome DevTools <https://developers.google.com/web/tools/chrome-devtools/>`_
or `Firefox Developer Tools <https://developer.mozilla.org/en-US/docs/Tools>`_, which can be
accessed by pressing F12. You can extend these with additional panels
`for React <https://reactjs.org/blog/2019/08/15/new-react-devtools.html>`_
and `for Redux <https://github.com/reduxjs/redux-devtools>`_.

These tools can be used for inspecting all frontend-related processes, from network traffic
up to rendering behavior. Take note of the `/api/events/` websocket connection, where all
asynchronous notification and results will be transferred.

Note that you should always debug using the development build of the GUI, using `npm start`,
as described in :ref:`the contributing section <building the client>`. Otherwise the debugging
experience may be painful, for example worse error output from react, minified source and
minified component names, ...

Debugging the API server
------------------------

If the API server returns a server error (500), the detailed exception should be logged
in the output of `libertem-server`. You can also try
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

If the problem is only reproducible using the default executor, you will have to follow the
`debugging instructions of dask-distributed <https://docs.dask.org/en/latest/debugging.html>`_.
As the API server can't use the synchronous :class:`~libertem.executor.inline.InlineJobExecutor`,
this is also the case when debugging problems that only occur in context of the API server.
