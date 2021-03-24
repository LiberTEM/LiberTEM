class UseDiscouragedWarning(FutureWarning):
    """
    This warning is thrown when a problematic feature of an API is used,
    which we currently don't have plans to remove (for backwards-compatability)
    """
    pass
