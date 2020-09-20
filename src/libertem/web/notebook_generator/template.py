import autopep8
from string import Template


class TemplateBase:
    '''
    Base class for template strings.

    Commonly used code template are available here.
    This can be override by specific code in subclasses.
    '''

    temp_ds = ['params = $params',
               'ds = ctx.load("$type", **params)']

    temp_dep = ["import matplotlib.pyplot as plt",
                "import libertem.api as lt",
                "import numpy as np"]

    temp_conn = ['cluster = executor.dask.DaskJobExecutor.connect("$conn_url")',
                 "ctx = lt.Context(executor=cluster)"]

    temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                     "${short}_result = ctx.run(${short}_analysis, progress=True)"]

    temp_save = ["np.save('${short}_result.npy', ${short}_result['intensity'])"]

    def code_formatter(self, code):
        return autopep8.fix_code(code)

    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)
