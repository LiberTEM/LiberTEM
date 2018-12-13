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
