Tips and tricks
===============

.. _`ssh forwarding`:

Using SSH Forwarding
--------------------

To securely connect to a remote instance of LiberTEM, you can use SSH forwarding. For example with conda:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 <remote-hostname> "source activate libertem; libertem-server"

Or, with virtualenv:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 <remote-hostname> "/path/to/virtualenv/bin/libertem-server"

This makes LiberTEM, which is running on `remote-hostname`, available on your local host via http://localhost:9000/


Running LiberTEM from an embedded interpreter
---------------------------------------------

If LiberTEM is run from within an embedded interpreter, the following steps should be taken. This is necessary for Python scripting in Digital Micrograph, for example.

The variable :code:`sys.argv` `may not be set in embedded interpreters <https://bugs.python.org/issue32573>`_, but it is expected by the :code:`multiprocessing` module when spawning new processes. This workaround guarantees that :code:`sys.argv` is set `until this is fixed upstream <https://github.com/python/cpython/pull/12463>`_:

.. code-block:: python
    
    if not hasattr(sys, 'argv'):
        sys.argv  = []


Furthermore, the `correct executable for spawning subprocesses <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_executable>`_ has to be set. 

.. code-block:: python
    
    multiprocessing.set_executable(
        os.path.join(sys.exec_prefix, 'pythonw.exe'))  # Windows only
