import logging
from typing import TYPE_CHECKING

import typing_extensions

from .models import AnalysisInfo, AnalysisResultInfo, CompoundAnalysisInfo

if TYPE_CHECKING:
    from .state import SharedState


log = logging.getLogger(__name__)


class RPCContext:
    """
    A high-level interface of methods available to RPC procedures.

    An RPCContext is always directly connected to a specific compound analysis,
    but it can access other analyses, too.

    Under the hood, this connects the RPC procedure to the current
    server-side state of the web API, and allows access to information
    about analyses, and allows to run an analysis.
    """
    def __init__(self, state: "SharedState", compound_analysis_id: str) -> None:
        self.state = state
        self.compound_analysis_id = compound_analysis_id

    def get_compound_analysis(self) -> CompoundAnalysisInfo:
        """
        Get information about the compound analysis in question.

        Returns
        -------
        CompoundAnalysisInfo
            Information about the analysis which this RPC is in connection with.
        """
        return self.state.compound_analysis_state[self.compound_analysis_id]

    def have_analysis_results(self, analysis_id: str) -> bool:
        return self.state.analysis_state.have_results(analysis_id)

    def get_analysis_details(self, analysis_id: str) -> AnalysisInfo:
        return self.state.analysis_state[analysis_id]

    def get_analysis_results(
        self, analysis_id: str
    ) -> AnalysisResultInfo:
        results = self.state.analysis_state.get_results(analysis_id)
        return results

    def run_analysis(self, analysis_id: str) -> None:
        raise NotImplementedError()


class ProcedureProtocol(typing_extensions.Protocol):
    def __call__(self, rpc_context: RPCContext):
        """
        The server-side interface that each procedure needs to implement

        Parameters
        ----------
        rpc_context : RPCContext
            Information about the related compound analysis is contained in this
            instance of :code:`RPCContext`, including methods to run analyses.
        """
        ...
