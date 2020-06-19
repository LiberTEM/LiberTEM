from string import Template
from libertem.analysis.base import Analysis


class CodeTemplate():
    def __init__(self, connection, dataset, compound_analysis):
        self.conn = connection
        self.ds = dataset
        self.compound_analysis = compound_analysis

    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)

    def dataset(self):
        ds_type = self.ds['type']
        ds_params = self.ds['params']
        temp_ds = ['params = $params',
                   'ds = ctx.load("$type", **params)']
        data = {'type': ds_type, 'params': ds_params}
        return self.format_template(temp_ds, data)

    def dependency(self):
        temp_dep = ["import matplotlib.pyplot as plt",
                    "import libertem.api as lt",
                    "import numpy as np",
                    "from libertem.analysis.getroi import get_roi",
                    "import numpy as np"]

        return '\n'.join(temp_dep)

    def initial_setup(self):
        return "%matplotlib nbagg"

    def connection(self):
        docs = ["# Connection"]
        if (self.conn['type'] == "cluster"):
            docs.append("***more about cluster conn")
            temp_conn = ['cluster = executor.dask.DaskJobExecutor.connect("$conn_url")',
                         "ctx = lt.Context(executor=cluster)"]
            data = {'conn_url': self.conn['url']}
            ctx = self.format_template(temp_conn, data)
            docs = '\n'.join(docs)
            return ctx, docs
        else:
            docs.append("**more about local conn**")
            ctx = "ctx = lt.Context()"
            docs = '\n'.join(docs)
            return ctx, docs

    def analysis(self):

        form_analysis = []

        for analysis in self.compound_analysis:

            type = analysis['analysisType']
            params = analysis['parameters']
            cls = Analysis.get_analysis_by_type(type)
            helperCls = cls.get_template_helper()
            helper = helperCls(params)

            plot_ = helper.get_plot()
            analy_ = helper.get_analysis()
            docs_ = helper.get_docs()

            form_analysis.append((docs_, analy_, plot_))

        return form_analysis
