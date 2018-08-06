Contributing
============

TODO

Running the Tests
-----------------

Our tests are written using pytest. For running them in a repeatable manner, we are using tox.
Tox automatically manages virtualenvs and allows testing on different Python versions and interpreter
implementations.

This makes sure that you can run the tests locally the same way as they are run in continuous integration.

After `installing tox <https://tox.readthedocs.io/en/latest/install.html>`_, you can run the tests on
all Python versions by simply running tox:

.. code-block:: shell

    $ tox

Or specify a specific environment you want to run:

.. code-block:: shell

    $ tox -e py36

Code Style
----------

TODO

 * pep8


Building the Documentation
--------------------------

Documentation building is also done with tox, see above for the basics.
To start the live building process:

.. code-block:: shell

    $ tox -e docs

You can then view a live-built version at http://localhost:8008

To build the HTML docs once:

.. code-block:: shell

    $ tox -e docs -- html
