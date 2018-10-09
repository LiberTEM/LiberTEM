class ExecutorError(Exception):
    pass


class JobCancelledError(Exception):
    """
    raised by async executors in run_job if the job was cancelled
    """
    pass


class JobExecutor(object):
    def run_job(self, job):
        raise NotImplementedError()

    def close(self):
        """
        cleanup resources used by this executor, if any
        """


class AsyncJobExecutor(object):
    async def run_job(self, job):
        raise NotImplementedError()

    async def close(self):
        pass

    async def cancel_job(self, job):
        pass
