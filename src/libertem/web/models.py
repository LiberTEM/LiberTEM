"""
Types and objects used in the web API
"""

from typing import TYPE_CHECKING, NamedTuple, Optional

from typing_extensions import TypedDict


if TYPE_CHECKING:
    from libertem.analysis.base import AnalysisResultSet
    from libertem.udf.base import UDFResults
    from libertem.io.dataset.base import DataSet


class JobInfo(TypedDict):
    id: str
    analysis: str
    dataset: str


class SerializedJobInfo(TypedDict):
    id: str
    analysis: str


AnalysisParameters = dict


class AnalysisDetails(TypedDict):
    analysisType: str
    parameters: AnalysisParameters


class AnalysisInfo(TypedDict):
    dataset: str
    analysis: str
    jobs: list[str]
    details: AnalysisDetails


class AnalysisResultInfo(NamedTuple):
    details: AnalysisDetails
    results: "AnalysisResultSet"
    job_id: str
    udf_results: Optional["UDFResults"]


class CompAnalysisDetails(TypedDict):
    mainType: str
    analyses: list[str]


class CompoundAnalysisInfo(TypedDict):
    dataset: str
    compoundAnalysis: str
    details: CompAnalysisDetails


class DatasetInfo(TypedDict):
    dataset: "DataSet"
    params: dict
    converted: dict
