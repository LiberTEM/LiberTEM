import tornado
import logging
from .notebook_generator.notebook_generator import notebook_generator
from .notebook_generator.copy import copy_notebook

from .state import SharedState

log = logging.getLogger(__name__)


class DownloadScriptHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self, compoundUuid: str):
        compoundAnalysis = self.state.compound_analysis_state[compoundUuid]
        analysis_ids = compoundAnalysis['details']['analyses']
        ds_id = self.state.analysis_state[analysis_ids[0]]['dataset']
        ds = self.state.dataset_state.datasets[ds_id]
        dataset = {
            "type": ds["params"]["params"]["type"],
            "params": ds['converted']
        }
        main_type = compoundAnalysis['details']['mainType'].lower()
        ds_name = ds["params"]["params"]["name"]

        analysis_details = []
        for id in analysis_ids:
            analysis_details.append(self.state.analysis_state[id]['details'])
        conn = self.state.executor_state.get_cluster_params()
        buf = notebook_generator(conn, dataset, analysis_details)
        self.set_header('Content-Type', 'application/vnd.jupyter.cells')
        self.set_header(
            'Content-Disposition',
            f'attachment; filename="{main_type}_{ds_name}.ipynb"',
        )
        self.write(buf.getvalue())


class CopyScriptHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self, compoundUuid: str):
        compoundAnalysis = self.state.compound_analysis_state[compoundUuid]
        analysis_ids = compoundAnalysis['details']['analyses']
        ds_id = self.state.analysis_state[analysis_ids[0]]['dataset']
        ds = self.state.dataset_state.datasets[ds_id]
        dataset = {
            "type": ds["params"]["params"]["type"],
            "params": ds['converted']
        }

        analysis_details = []
        for id in analysis_ids:
            analysis_details.append(self.state.analysis_state[id]['details'])
        conn = self.state.executor_state.get_cluster_params()
        notebook = copy_notebook(conn, dataset, analysis_details)
        self.write(notebook)
