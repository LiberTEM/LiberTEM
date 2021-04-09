import importlib
from importlib import metadata

KEY = 'libertem.plugins.v1'


def load_plugins():
    entry_points = metadata.entry_points()
    if KEY not in entry_points:
        return {}
    plugins = {}
    for ep in entry_points[KEY]:
        if ':' not in ep.value:
            continue  # FIXME: emit warning?
        modname, classname = ep.value.split(':')
        mod = importlib.import_module(modname)
        klass = getattr(mod, classname)
        plugins[(modname, classname)] = klass

    return plugins
