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
