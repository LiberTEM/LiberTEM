import tornado
import logging

from .state import SharedState

log = logging.getLogger(__name__)


class DownloadScriptHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self, compoundUuid: str):
        compoundAnalysis = self.state.compound_analysis_state[compoundUuid]
        analysisList = compoundAnalysis['details']['analyses']

        for analysis_id in analysisList:
            print(self.state.analysis_state[analysis_id]['details'])
