import pytest
import numpy as np

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

    with pytest.raises(AttributeError) as einfo:
        results.foo
    assert einfo.match("not found")
    assert einfo.match("have: test")

    assert results.test == result
    assert len(results) == 1
    assert results[0] == result
    for item in results:
        assert item == result


def test_result_coerce_to_array():
    result = AnalysisResult(
        raw_data=(np.zeros((16, 16)), np.ones((16, 16))),
        visualized=None,
        title="test",
        desc="test",
        key="test"
    )

    # "object __array__ method not producing an array"
    assert np.array(result).shape == (2, 16, 16)
