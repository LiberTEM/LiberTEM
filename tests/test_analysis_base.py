import pytest

from libertem.analysis.base import AnalysisResultSet, AnalysisResult


def test_result_set():
    result = AnalysisResult(
        raw_data=None,
        visualized=None,
        title="test",
        desc="test",
        key="test"
    )

    results = AnalysisResultSet([result])

    with pytest.raises(AttributeError) as e:
        results.foo
    assert "not found" in str(e)
    assert "have: test" in str(e)

    assert results.test == result
    assert len(results) == 1
    assert results[0] == result
