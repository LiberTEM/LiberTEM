import os
import stat

import tornado.web

from libertem.io.fs import get_fs_listing, FSError, stat_path
from .messages import Message
from .state import SharedState


class LocalFSBrowseHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        executor = await self.state.executor_state.get_executor()
        path = self.request.arguments['path']
        assert len(path) == 1
        path = path[0].decode("utf8")
        try:
            listing = await executor.run_function(get_fs_listing, path)
            msg = Message().directory_listing(
                **listing
            )
            self.write(msg)
        except FSError as e:
            msg = Message().browse_failed(
                path=path,
                code=e.code,
                msg=str(e),
                alternative=e.alternative,
            )
            self.write(msg)


class LocalFSStatHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry) -> None:
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        executor = await self.state.executor_state.get_executor()
        path = self.request.arguments['path']
        assert len(path) == 1
        path = path[0].decode("utf8")
        try:
            stat_result = await executor.run_function(stat_path, path)
            if stat.S_ISDIR(stat_result.st_mode) and not path.endswith(os.path.sep):
                path_slash = path + os.path.sep
            else:
                path_slash = path
            msg = Message().browse_stat_result(
                path=path,
                dirname=os.path.dirname(path_slash),
                basename=os.path.basename(path_slash),
                stat_result=stat_result,
            )
            self.write(msg)
        except FSError as e:
            msg = Message().stat_failed(
                path=path,
                code=e.code,
                msg=str(e),
                alternative=e.alternative,
            )
            self.write(msg)
