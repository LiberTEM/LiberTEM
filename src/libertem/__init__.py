# keep version and revision in separate files such that we can
# generate them without any regex magic

from .__version__ import __version__

try:
    from ._baked_revision import revision
except ImportError:
    from .versioning import get_git_rev
    revision = get_git_rev()

__all__ = [
    "revision", "__version__"
]
