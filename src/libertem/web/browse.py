import tornado.web

from libertem.io.fs import get_fs_listing, FSError
from .messages import Message
from .state import SharedState


class LocalFSBrowseHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        executor = self.state.executor_state.get_executor()
        path = self.request.arguments['path']
        assert len(path) == 1
        path = path[0].decode("utf8")
        try:
            listing = await executor.run_function(get_fs_listing, path)
            msg = Message(self.state).directory_listing(
                **listing
            )
            self.write(msg)
        except FSError as e:
            msg = Message(self.state).browse_failed(
                path=path,
                code=e.code,
                msg=str(e),
                alternative=e.alternative,
            )
            self.write(msg)
