Running distributed tests
=========================

To test the behavior of the distributed parts of LiberTEM, there are now a
few automated test cases available. Because of the distributed nature of the
tests, they need to be run across multiple network nodes (computers, VMs, or
containers). We ship a :code:`docker-compose` configuration to spin up a
scheduler and two workers as separate docker containers.

Requirements
------------

To run the distributed tests, you need to install (a current version of) :code:`docker`
and :code:`docker-compose`. See
`the official docker documentation <https://docs.docker.com/install/>`_ to get started.
To start the test environment, change to the :code:`packaging/docker/` directory and
run the following:

.. code:: shell

   $ docker-compose up

Note that you need to run the above command as a user that can access the docker daemon,
for example by being in the :code:`docker` group. The containers are running as long as
the :code:`docker-compose` command is running, so you can stop them using :code:`Ctrl-C`.

You can then run the distributed tests using (from the repository root directory):

.. code:: shell

   $ DASK_SCHEDULER_ADDRESS=tcp://localhost:8786 pytest -m dist tests/

Or via :code:`tox`:

.. code:: shell

   $ DASK_SCHEDULER_ADDRESS=tcp://localhost:8786 tox -e py37 -- -m dist tests/

After changing LiberTEM code, you need to rebuild and restart the containers. This can be
done by running:

.. code:: shell

   $ docker-compose build && docker-compose up

Or, as one command:

.. code:: shell

   $ docker-compose up --build

The rebuild should be faster than the initial build, which is accomplished by careful
use of the layer caching feature of docker. This also menas that you need to update
the :code:`packaging/docker/requirements.txt` file by running the provided
script :code:`update_reqs.sh` when the dependencies of LiberTEM change, or when new
versions of dependencies are released.
