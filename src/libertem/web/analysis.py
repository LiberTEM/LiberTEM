import tornado.web

from .base import CORSMixin, log_message
from .messages import Message


class AnalysisDetailHandler(CORSMixin, tornado.web.RequestHandler):
    """
    API Handler for CRUD of analyses.

    Each Analysis has a connection to a Dataset, and
    zero or one Job.

    """
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def put(self, uuid):
        """
        Register or update an analysis, which is then later associated
        to their results.

        request body contains the analysis type and parameters

        On update, we let the currently running jobs continue running,
        jobs are handled separately by the JobDetailHandler
        """
        request_data = tornado.escape.json_decode(self.request.body)
        details = request_data['analysis']
        params = details['analysis']['parameters']
        analysis_type = details['analysis']['type']
        dataset_id = details["dataset"]
        existing_analysis = self.data.get_analysis(uuid)
        if existing_analysis is None:
            return await self._create_analysis(uuid, dataset_id, analysis_type, params)
        else:
            return await self._update_analysis(uuid, existing_analysis, params)

    async def _create_analysis(self, uuid, dataset_id, analysis_type, params):
        self.data.create_analysis(uuid, dataset_id, analysis_type, params)
        msg = Message(self.data).create_analysis(uuid, dataset_id, analysis_type, params)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def _update_analysis(self, uuid, existing_analysis, params):
        self.data.update_analysis(uuid, params)
        msg = Message(self.data).update_analysis(uuid, params)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def delete(self, uuid):
        """
        Remove an analysis, stop all related jobs and remove
        analysis results.
        """
        result = await self.data.remove_analysis(uuid)
        if result:
            msg = Message(self.data).analysis_removed(uuid)
        else:
            msg = Message(self.data).analysis_removal_failed(uuid)
        log_message(msg)
