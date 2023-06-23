class UDFException(Exception):
    """
    Raised when the UDF interface is somehow misused
    """
    pass


class ExecutorSpecException(Exception):
    """
    Raised when there is an error specifying an
    Executor or its resources / workers
    """
    pass


class UDFRunCancelled(Exception):
    """
    Raised when the UDF run was cancelled, either when the job was cancelled
    using :meth:`AsyncJobExecutor.cancel`, or when the underlying data source
    was interrupted.
    """
