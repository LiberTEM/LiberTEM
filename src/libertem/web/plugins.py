import tornado.web

from libertem.plugins.v1.discover import load_plugins
from libertem.udf.base import UDFRunner
from .base import CORSMixin, log_message
from .messages import Message
from .state import SharedState


class PluginListHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.dataset_state = state.dataset_state
        self.event_registry = event_registry

    def get_plugin_details(self, key, plugin, dataset):
        udf_class = plugin().get_udf_class()

        # XXX FIXME XXX
        # we only have the UDF class, we can't instantiate it
        # because we don't know the right parameters
        # so we have a problem finding the available channels!
        buffers = UDFRunner.inspect_udf(udf_class, dataset)

        return {
            'title': udf_class.title,
            'channels': [
                {
                    "name": name,
                    "kind": channel.kind,
                }
                for name, channel in channels.items()
            ],
            'id': f"{key[0]}:{key[1]}",
        }

    def get_plugins(self, dataset):
        plugins = load_plugins()
        result = {}
        for key, plugin in plugins.items():
            details = self.get_plugin_details(key, plugin, dataset)
            result[details['id']] = details
        return result

    async def get(self, dataset_uuid):
        try:
            dataset = self.dataset_state[dataset_uuid]
        except KeyError:
            self.set_status(404, "dataset with uuid %s not found" % dataset_uuid)
            return
        msg = Message(self.state).plugin_list(self.get_plugins(dataset))
        log_message(msg)
        self.write(msg)
