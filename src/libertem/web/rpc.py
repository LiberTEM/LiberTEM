import logging
from typing import TYPE_CHECKING, Callable, TypeVar, Union

import uuid
import typing_extensions

from libertem.analysis.base import AnalysisResultSet

from .models import AnalysisInfo, AnalysisResultInfo, CompoundAnalysisInfo

if TYPE_CHECKING:
    from .state import SharedState
    from .engine import JobEngine


log = logging.getLogger(__name__)

T = TypeVar('T')


class RPCContext:
    """
    A high-level interface of methods available to RPC procedures.

    An RPCContext is always directly connected to a specific compound analysis,
    but it can access other analyses, too.

    Under the hood, this connects the RPC procedure to the current
    server-side state of the web API, and allows access to information
    about analyses, and allows to run an analysis.
    """
    def __init__(
        self, state: "SharedState", compound_analysis_id: str, engine: "JobEngine"
    ) -> None:
        self.state = state
        self.compound_analysis_id = compound_analysis_id
        self.engine = engine

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

    async def run_analysis(self, analysis_id: str) -> AnalysisResultSet:
        job_id = str(uuid.uuid4())
        return await self.engine.run_analysis(analysis_id, job_id)

    async def run_sync(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """
        To run a blocking, more compute-intensive function as part of
        a RPC, please wrap the call with this function!
        """
        return await self.engine.run_sync(fn, *args, **kwargs)


class SyncProcedureProtocol(typing_extensions.Protocol):
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


class AsyncProcedureProtocol(typing_extensions.Protocol):
    async def __call__(self, rpc_context: RPCContext):
        """
        The server-side interface that each procedure needs to implement (async version)

        Parameters
        ----------
        rpc_context : RPCContext
            Information about the related compound analysis is contained in this
            instance of :code:`RPCContext`, including methods to run analyses.
        """
        ...


ProcedureProtocol = Union[AsyncProcedureProtocol, SyncProcedureProtocol]
