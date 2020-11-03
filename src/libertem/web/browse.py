import tornado.web
from libertem.io.fs import FSError, get_fs_listing

from .base import SessionsHandler
from .messages import Message
from .state import SharedState


class LocalFSBrowseHandler(SessionsHandler, tornado.web.RequestHandler):
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
