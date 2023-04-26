Video stream from server to client
==================================

This prototype generates a stream from a changing NumPy array and shows it in the browser.

It is based on the aiortc examples in https://github.com/aiortc/aiortc/tree/main/examples

Running
-------

First install the required packages:

.. code-block:: console

    $ pip install aiohttp aiortc

When you start the example, it will create an HTTP server which you
can connect to from your browser:

.. code-block:: console

    $ python server.py

You can then browse to the following page with your browser:

http://127.0.0.1:8080

Once you click `Start`, the browser will start receiving a video stream from the server.

Additional options
------------------

If you want to enable verbose logging, run:

.. code-block:: console

    $ python server.py -v
