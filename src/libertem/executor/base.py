class JobExecutor(object):
    def run_job(self, job):
        raise NotImplementedError()

    def close(self):
        """
        cleanup resources used by this executor, if any
        """
