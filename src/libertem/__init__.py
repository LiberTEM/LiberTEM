__version__ = "0.1.0.dev0"

try:
    from ._baked_revision import revision
except ImportError:
    from .versioning import get_git_rev
    revision = get_git_rev()
