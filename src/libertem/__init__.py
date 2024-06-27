try:
    from .__version__ import __version__, __version_tuple__
    revision = __version_tuple__[-1]
except ModuleNotFoundError:
    __version__ = "0.0.0.dev0"
    from .versioning import get_git_rev
    revision = get_git_rev()


__all__ = [
    "revision", "__version__"
]
