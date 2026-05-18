Running distributed tests
=========================

To test the behavior of the distributed parts of LiberTEM, there are now a
few automated test cases available. Because of the distributed nature of the
tests, they need to be run across multiple network nodes (computers, VMs, or
containers). We ship a :code:`docker compose` configuration to spin up a
scheduler and two workers as separate docker containers.

Requirements
------------

To run the distributed tests, you need to install (a current version of) :code:`docker`
and :code:`docker compose`. See
`the official docker documentation <https://docs.docker.com/get-docker/>`_ to get started.
To start the test environment, change to the :code:`packaging/docker/` directory and
run the following:

.. code:: shell

   $ bash test.sh

Note that you need to run the above command as a user that can access the docker
daemon, for example by being in the :code:`docker` group. As we are running the
tests in a docker container, the environment of the `tests` container will
automatically match the environment of the workers.

After changing LiberTEM code, you need to rebuild and restart the containers. Just re-run
:code:`test.sh` to start from the top.

The rebuild should be faster than the initial build, which is accomplished by careful
use of the layer caching feature of docker.
