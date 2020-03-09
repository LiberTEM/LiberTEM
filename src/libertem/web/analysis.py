import io

import tornado.web

from .base import CORSMixin, log_message
from .messages import Message
from .state import SharedState


class AnalysisDetailHandler(CORSMixin, tornado.web.RequestHandler):
    """
    API Handler for CRUD of analyses.

    Each Analysis has a connection to a Dataset, and
    zero or one Job.

    """
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def put(self, uuid):
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
        details = request_data['analysis']
        params = details['parameters']
        analysis_type = details['type']
        existing_analysis = self.state.analysis_state.get(uuid)
        if existing_analysis is None:
            return await self._create_analysis(uuid, dataset_id, analysis_type, params)
        else:
            return await self._update_analysis(
                uuid, dataset_id, analysis_type, existing_analysis, params
            )

    async def _create_analysis(self, uuid, dataset_id, analysis_type, params):
        self.state.analysis_state.create(uuid, dataset_id, analysis_type, params)
        msg = Message(self.state).create_analysis(uuid, dataset_id, analysis_type, params)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def _update_analysis(self, uuid, dataset_id, analysis_type, existing_analysis, params):
        self.state.analysis_state.update(uuid, analysis_type, params)
        msg = Message(self.state).update_analysis(uuid, dataset_id, analysis_type, params)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def delete(self, uuid):
        """
        Remove an analysis, stop all related jobs and remove
        analysis results.
        """
        result = self.state.analysis_state.remove(uuid)
        if result:
            msg = Message(self.state).analysis_removed(uuid)
        else:
            # FIXME: concrete error message?
            msg = Message(self.state).analysis_removal_failed(uuid, "analysis could not be removed")
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

    async def get(self, uuid):
        details, results = self.state.analysis_state.get_results(uuid)
        import h5py

        bio = io.BytesIO()
        with h5py.File(bio) as f:
            for k in results.keys():
                f[k] = results[k]

        # FIXME: stream file (maybe temporary file w/ sendfile?), correct content-type
        # FIXME: add parameters to h5 file
        self.set_header('Content-Type', 'application/x-hdf5')
        self.set_header('Content-Disposition', 'attachment; filename="results.h5"')
        self.write(bio.getvalue())
