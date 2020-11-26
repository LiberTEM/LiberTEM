# keep version and revision in separate files such that we can
# generate them without any regex magic

import sys
from .__version__ import __version__

try:
    from ._baked_revision import revision
except ImportError:
    from .versioning import get_git_rev
    revision = get_git_rev()

__all__ = [
    "revision", "__version__"
]


def adjust_event_loop_policy():
    import asyncio
    # stolen from ipykernel:
    if sys.platform.startswith("win") and sys.version_info >= (3, 8):
        try:
            from asyncio import (
                WindowsProactorEventLoopPolicy,
                WindowsSelectorEventLoopPolicy,
            )
        except ImportError:
            # not affected
            pass
        else:
            # FIXME this might fail if the event loop policy has been overridden by something
            # incompatible that is not a WindowsProactorEventLoopPolicy. Dask.distributed creates
            # a custom event loop policy, for example. Fortunately that one should be compatible.
            # Better would be "duck typing", i.e. to check if the required add_reader() method
            # of the loop throws a NotImplementedError.
            if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                # WindowsProactorEventLoopPolicy is not compatible with tornado 6
                # fallback to the pre-3.8 default of Selector
                asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())


adjust_event_loop_policy()
