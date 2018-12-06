import subprocess
import json
from time import sleep

import tornado.util
from dask import distributed as dd
from distributed.asyncio import AioClient
from .base import JobExecutor, AsyncJobExecutor, JobCancelledError


# NOTE:
# if you are mistakenly using dd.Client in an asyncio environment,
# you get a message like this:
# error message: "RuntimeError: Non-thread-safe operation invoked on an event loop
# other than the current one"
# related: debugging via env var PYTHONASYNCIODEBUG=1


class CommonDaskMixin(object):
    def _get_futures(self, job):
        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            locations = task.get_locations()
            if locations is not None and len(locations) == 0:
                raise ValueError("no workers found for task")
            submit_kwargs['workers'] = locations
            futures.append(
                self.client.submit(task, **submit_kwargs)
            )
        return futures


class AsyncDaskJobExecutor(CommonDaskMixin, AsyncJobExecutor):
    def __init__(self, client, is_local=False):
        self.is_local = is_local
        self.client = client
        self._futures = {}

    async def close(self):
        await self.client.close()
        if self.is_local:
            try:
                self.client.cluster.close(timeout=1)
            except tornado.util.TimeoutError:
                pass

    async def run_job(self, job):
        futures = self._get_futures(job)
        self._futures[job] = futures
        async for future, result in dd.as_completed(futures, with_results=True):
            if future.cancelled():
                raise JobCancelledError()
            yield result
        del self._futures[job]

    async def cancel_job(self, job):
        if job in self._futures:
            futures = self._futures[job]
            await self.client.cancel(futures)

    @classmethod
    async def connect(cls, scheduler_uri, *args, **kwargs):
        """
        Connect to remote dask scheduler

        Returns
        -------
        AsyncDaskJobExecutor
            the connected JobExecutor
        """
        client = await AioClient(address=scheduler_uri)
        return cls(client=client, *args, **kwargs)

    @classmethod
    async def make_local(cls, cluster_kwargs=None, client_kwargs=None):
        """
        Spin up a local dask cluster

        interesting cluster_kwargs:
            threads_per_worker
            n_workers

        Returns
        -------
        AsyncDaskJobExecutor
            the connected JobExecutor
        """
        cluster = dd.LocalCluster(**(cluster_kwargs or {}))
        client = await AioClient(cluster, **(client_kwargs or {}))
        return cls(client=client, is_local=True)


class DaskJobExecutor(CommonDaskMixin, JobExecutor):
    def __init__(self, client, is_local=False, subprocess=None):
        self.is_local = is_local
        self.client = client
        self.subprocess = subprocess

    def run_job(self, job):
        futures = self._get_futures(job)
        for future, result in dd.as_completed(futures, with_results=True):
            yield result

    def close(self):
        if self.is_local:
            if self.client.cluster is not None:
                try:
                    self.client.cluster.close(timeout=1)
                except tornado.util.TimeoutError:
                    pass
        if self.subprocess is not None:
            self.subprocess.terminate()
        self.client.close()

    @classmethod
    def connect(cls, scheduler_uri, *args, **kwargs):
        """
        Connect to a remote dask scheduler

        Returns
        -------
        DaskJobExecutor
            the connected JobExecutor
        """
        client = dd.Client(address=scheduler_uri)
        return cls(client=client, *args, **kwargs)

    @classmethod
    def make_local(cls, cluster_kwargs=None, client_kwargs=None):
        """
        Spin up a local dask cluster

        interesting cluster_kwargs:
            threads_per_worker
            n_workers

        Returns
        -------
        DaskJobExecutor
            the connected JobExecutor
        """
        cluster = dd.LocalCluster(**(cluster_kwargs or {}))
        client = dd.Client(cluster, **(client_kwargs or {}))
        return cls(client=client, is_local=True)

    @classmethod
    def subprocess_make_local(cls, cluster_kwargs=None, client_kwargs=None):
        c = ("# breakme\n"
            "import sys\n"
            "from time import sleep\n"
            "import json\n"

            "try:\n"
            "    import distributed as dd\n"

            "    input = sys.stdin.readline()\n"
            "    cluster_kwargs = json.loads(input)\n"
            "#    cluster_kwargs['breakme'] = 'die die die'\n"
            "    cluster = dd.LocalCluster(**(cluster_kwargs or {}))\n"
            "    response = {'scheduler_address': cluster.scheduler_address, 'success': True}\n"
            "    print(json.dumps(response), file=sys.stdout, flush=True)\n"
            "    while True:\n"
            "        sleep(100)\n"
            "except Exception as e:\n"
            "    response = {'success': False, 'exception': str(e)}\n"
            "    print(json.dumps(response), file=sys.stdout, flush=True)\n"
            "    raise\n")
        # We trust that the environment is set up
        # to start the correct python interpreter
        try:
            # On Windows use pythonw.exe
            sp = subprocess.Popen(
                ['pythonw', '-c', c],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
        except FileNotFoundError:
            # Fall back to python / python.exe
            sp = subprocess.Popen(
                ['python', '-c', c],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
        print(json.dumps(cluster_kwargs), file=sp.stdin, flush=True)
        # wait for syntax error or startup failures
        sleep(1)
        # Terminated prematurely
        if sp.poll() is not None:
            stderr = sp.stderr.read()
            stdout = sp.stdout.read()
            raise Exception(
                "Starting subprocess failed. stderr: %s\n\nstdout: %s" % (stderr, stdout)
            )
        # We made sure that the process either terminates or writes something to stdout
        # so that this doesn't block forever
        response = json.loads(sp.stdout.readline())
        # print(response)
        if not response['success']:
            sp.terminate()
            stderr = sp.stderr.read()
            stdout = sp.stdout.read()
            raise Exception("Starting subprocess failed. Exception: %s\n\nstderr: %s\n\n stdout: %s"
                % (response['exception'], stderr, stdout)
            )
        uri = response['scheduler_address']
        # print(uri)
        client = dd.Client(uri, **(client_kwargs or {}))
        return cls(client=client, is_local=True, subprocess=sp)
