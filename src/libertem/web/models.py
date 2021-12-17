"""
Types and objects used in the web API
"""

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

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


AnalysisParameters = Dict


class AnalysisDetails(TypedDict):
    analysisType: str
    parameters: AnalysisParameters


class AnalysisInfo(TypedDict):
    dataset: str
    analysis: str
    jobs: List[str]
    details: AnalysisDetails


class AnalysisResultInfo(NamedTuple):
    details: AnalysisDetails
    results: "AnalysisResultSet"
    job_id: str
    udf_results: Optional["UDFResults"]


class CompAnalysisDetails(TypedDict):
    mainType: str
    analyses: List[str]


class CompoundAnalysisInfo(TypedDict):
    dataset: str
    compoundAnalysis: str
    details: CompAnalysisDetails


class DatasetInfo(TypedDict):
    dataset: "DataSet"
    params: Dict
    converted: Dict
