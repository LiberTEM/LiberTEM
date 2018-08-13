Usage
=====

.. include:: _single_node.rst

Starting the LiberTEM Server
----------------------------

LiberTEM is based on a client-server architecture. To use LiberTEM, you need to
have the server running on the machine where your data is available.

After :doc:`installing LiberTEM <install>`, activate the virtualenv or conda environment.

You can then start the LiberTEM server by running:

.. code-block:: shell

    (libertem) $ libertem-server

By default, this starts the server on http://localhost:9000, which you can verify by the
log output::

    [2018-08-08 13:57:58,266] INFO [libertem.web.server.main:886] listening on localhost:9000

There are a few command line options available:: 

    Usage: libertem-server [OPTIONS]

    Options:
      --port INTEGER  port on which the server should listen on
      --host TEXT     host on which the server should listen on
      --help          Show this message and exit.
    
As there is currently no authentication yet, it is discouraged to listen on a different host than
`127.0.0.1` / `localhost`. As a workaround, if you want to access LiberTEM from a different computer,
you can use ssh port forwarding (example command for conda):

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 remote-hostname "source activate libertem; libertem-server"

Or, with virtualenv:

.. code-block:: shell

     $ ssh -L 9000:localhost:9000 remote-hostname "/path/to/virtualenv/bin/libertem-server"

This then makes LiberTEM that is running on `remote-host` available on your local host via http://localhost:9000/


The user interface
------------------

TODO
