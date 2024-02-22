import pytest

with pytest.warns(DeprecationWarning):
    import libertem.analysis.gridmatching as grm  # NOQA: 401
