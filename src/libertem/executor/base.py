class ExecutorError(Exception):
    pass


class JobCancelledError(Exception):
    """
    raised by async executors in run_job if the job was cancelled
    """
    pass


class JobExecutor(object):
    def run_job(self, job):
        """
        run a Job
        """
        raise NotImplementedError()

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
        """
        raise NotImplementedError()

    def close(self):
        """
        cleanup resources used by this executor, if any
        """

    def get_available_workers(self):
        """
        returns a list of dicts with available workers
        keys of the dictionary:
            name : the identifying name of the worker
            host : ip address or hostname where the worker is running
        """
        raise NotImplementedError()


class AsyncJobExecutor(object):
    async def run_job(self, job):
        """
        run a Job
        """
        raise NotImplementedError()

    async def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
        """
        raise NotImplementedError()

    async def close(self):
        """
        cleanup resources used by this executor, if any
        """

    async def cancel_job(self, job):
        """
        cancel execution of `job`
        """
        pass

    async def get_available_workers(self):
        raise NotImplementedError()
