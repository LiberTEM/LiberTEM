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

    temp_dep_conn = ["import distributed as dd",
                     "from libertem.executor.dask import DaskJobExecutor"]

    temp_conn = ['client = dd.Client("$conn_url")',
                 "executor = DaskJobExecutor(client)",
                 "ctx = lt.Context(executor=executor)"]

    def temp_conn_tcp(self):
        if self.type == 'notebook':
            return ['client = dd.Client("$conn_url")',
                    "executor = DaskJobExecutor(client)",
                    "ctx = lt.Context(executor=executor)"]
        else:
            indent = " "*4
            return ['if __name__ == "__main__":',
                    f'{indent}client = dd.Client("$conn_url")',
                    f'{indent}executor = DaskJobExecutor(client)',
                    f'{indent}with lt.Context(executor=executor) as ctx:']

    def temp_conn_local(self):
        if self.type == 'notebook':
            return ["ctx = lt.Context()"]
        else:
            indent = " "*4
            return ['if __name__ == "__main__":',
                    f"{indent}with lt.Context() as ctx:"]

    temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                     "${short}_result = ctx.run(${short}_analysis, progress=True)"]

    temp_save = ["np.save('${short}_result.npy', ${short}_result['intensity'])"]

    def code_formatter(self, code):
        return autopep8.fix_code(code)

    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)
