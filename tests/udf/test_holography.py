import pytest


def test_smoke():
    with pytest.raises(ImportError, match='The holography implementation is removed'):
        import libertem.udf.holography  # noqa: F401
