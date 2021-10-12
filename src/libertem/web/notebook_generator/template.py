import autopep8
from string import Template


class TemplateBase:
    '''
    Base class for template strings.

    Commonly used code template are available here.
    This can be override by specific code in subclasses.
    '''

    temp_ds_backend = ["io_backend = ${backend_cls}()"]
    temp_ds = ['params = $params',
               'ds = ctx.load("$type", **params)']

    temp_dep_ds = ["from libertem.io.dataset.base import ${backend_cls}"]

    temp_dep = ["import matplotlib.pyplot as plt",
                "import libertem.api as lt",
                "import numpy as np"]

    temp_dep_conn = ["import distributed as dd",
                     "from libertem.executor.dask import DaskJobExecutor"]

    temp_conn = ['client = dd.Client("$conn_url")',
                 "executor = DaskJobExecutor(client)",
                 "ctx = lt.Context(executor=executor)"]

    temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                     "${short}_result = ctx.run(${short}_analysis, progress=True)"]

    temp_save = ["np.save('${short}_result.npy', ${short}_result['intensity'])"]

    def code_formatter(self, code):
        return autopep8.fix_code(code)

    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)
