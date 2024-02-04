import io
import logging
import inspect

import tornado.web
from libertem.analysis.base import Analysis
from libertem.web.engine import JobEngine

from libertem.web.rpc import RPCContext

from .base import CORSMixin, log_message
from .messages import Message
from .state import SharedState
from libertem.io.writers.results.base import ResultFormatRegistry

logger = logging.getLogger(__name__)


class AnalysisDetailHandler(CORSMixin, tornado.web.RequestHandler):
    """
    API Handler for CRUD of analyses.

    Each Analysis has a connection to a Dataset, and
    zero or one Job.

    """
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def put(self, compoundUuid, uuid):
        """
        Register or update an analysis, which is then later associated
        to their results.

        request body contains the analysis type and parameters

        On update, we let the currently running jobs continue running,
        jobs are handled separately by the JobDetailHandler.

        Update can change both parameters and analysis type.
        """
        request_data = tornado.escape.json_decode(self.request.body)
        dataset_id = request_data["dataset"]
        details = request_data['details']
        params = details['parameters']
        analysis_type = details['analysisType']
        existing_analysis = self.state.analysis_state.get(uuid)
        if existing_analysis is None:
            return await self._create(uuid, dataset_id, analysis_type, params)
        else:
            return await self._update(
                uuid, dataset_id, analysis_type, existing_analysis, params
            )

    async def _create(self, uuid, dataset_id, analysis_type, params):
        self.state.analysis_state.create(uuid, dataset_id, analysis_type, params)
        msg = Message().create_analysis(uuid, dataset_id, analysis_type, params)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def _update(self, uuid, dataset_id, analysis_type, existing_analysis, params):
        self.state.analysis_state.update(uuid, analysis_type, params)
        msg = Message().update_analysis(uuid, dataset_id, analysis_type, params)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def delete(self, compoundUuid, uuid):
        """
        Remove an analysis, stop all related jobs and remove
        analysis results.
        """
        result = await self.state.analysis_state.remove(uuid)
        if result:
            msg = Message().analysis_removed(uuid)
        else:
            # FIXME: concrete error message?
            msg = Message().analysis_removal_failed(uuid, "analysis could not be removed")
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)


class DownloadDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    def _get_format(self):
        # FIXME: unused for now
        fmt = self.request.arguments['fmt']
        assert len(fmt) == 1
        return fmt[0].decode("utf8")

    async def get(self, compoundUuid: str, uuid: str, file_format_id: str):
        details, result_set, job_id, udf_results = self.state.analysis_state.get_results(uuid)
        format_cls = ResultFormatRegistry.get_format_by_id(file_format_id)
        result_formatter = format_cls(result_set)
        buf = io.BytesIO()
        result_formatter.serialize_to_buffer(buf)

        self.set_header('Content-Type', result_formatter.get_content_type())
        self.set_header(
            'Content-Disposition',
            'attachment; filename="%s"' % result_formatter.get_filename(),
        )
        # FIXME: stream file (maybe temporary file w/ sendfile?)
        self.write(buf.getvalue())


class CompoundAnalysisHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        dataset_id = request_data['dataset']
        details = request_data['details']
        main_type = details['mainType']
        analyses = details['analyses']
        created = self.state.compound_analysis_state.create_or_update(
            uuid, main_type, dataset_id, analyses
        )
        serialized = self.state.compound_analysis_state.serialize(uuid)
        if created:
            msg = Message().compound_analysis_created(serialized)
        else:
            msg = Message().compound_analysis_updated(serialized)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def delete(self, uuid):
        ca = self.state.compound_analysis_state[uuid]
        for analysis_id in ca["details"]["analyses"]:
            result = await self.state.analysis_state.remove(analysis_id)
            if result:
                msg = Message().analysis_removed(analysis_id)
            else:
                # FIXME: concrete error message?
                msg = Message().analysis_removal_failed(
                    analysis_id, "analysis could not be removed"
                )
            log_message(msg)
            self.event_registry.broadcast_event(msg)

        self.state.compound_analysis_state.remove(uuid)

        msg = Message().compound_analysis_removed(uuid)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)


class AnalysisRPCHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry
        self.engine = JobEngine(state, event_registry)

    async def put(self, compound_analysis_id: str, proc_name: str):
        rpc_context = RPCContext(
            state=self.state,
            compound_analysis_id=compound_analysis_id,
            engine=self.engine,
        )
        comp_ana = rpc_context.get_compound_analysis()
        ana_type = comp_ana['details']['mainType']
        analysis_cls = Analysis.get_analysis_by_type(ana_type)
        rpc_def = analysis_cls.get_rpc_definitions()
        if proc_name not in rpc_def:
            self.set_status(400, "Bad request: unknown RPC method")
            self.write({
                "status": "error",
                "msg": "unknown RPC method",
            })
            return

        ProcCls = rpc_def[proc_name]

        proc = ProcCls()
        # support both sync and async variants here:
        if inspect.iscoroutinefunction(proc.__call__):
            result = await proc(rpc_context)
        else:
            return proc(rpc_context)
        self.write(result)
