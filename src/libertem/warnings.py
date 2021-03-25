"""
Custom warning subclasses. Useful for selectively filtering. For example, to
cause an exception for all `UseDiscouragedWarning` that are thrown in the test
suite, one can run:

$ pytest -Werror:libertem.warnings.UseDiscouragedWarning tests/
"""


class UseDiscouragedWarning(FutureWarning):
    """
    This warning is thrown when a problematic feature of an API is used,
    which we currently don't have plans to remove (for backwards-compatability)
    """
    pass
