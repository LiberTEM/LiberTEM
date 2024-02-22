import pytest

try:
    with pytest.warns(DeprecationWarning):
        import libertem.analysis.fullmatch as fm  # NOQA: 401
except ModuleNotFoundError as e:
    fm = None
    missing = e.name
