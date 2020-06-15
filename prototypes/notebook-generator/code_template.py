from string import Template
from libertem.analysis.base import AnalysisRegistry


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

        temp_docs = ["# $analysis",
                     "**description**"]

        temp_roi = ["roi_params = $roi_params",
                    "roi = get_roi(roi_params, ds.shape.nav)"]

        temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                         "$roi",
                         "udf = ${short}_analysis.get_udf()",
                         "${short}_result = ctx.run_udf(ds, udf, roi, progress=True)"]

        form_analysis = []

        for analysis in self.compound_analysis:

            an_type = analysis['analysisType']
            an_params = analysis['parameters']
            cls = AnalysisRegistry.get_analysis_by_type(an_type)['class']
            helperCls = cls.get_template_helper()
            helper = helperCls()

            analysis_api = helper.api
            short = helper.short_name

            plot_ = helper.get_plot()
            docs_ = self.format_template(temp_docs, {'analysis': an_type})

            if 'roi' in an_params.keys():
                data = {'roi_params': an_params}
                roi = self.format_template(temp_roi, data)
            else:
                roi = "roi = None"

            params = helper.convert_params(an_params, 'ds')

            data = {'short': short, 'analysis_api': analysis_api, 'params': params, 'roi': roi}
            analy_ = self.format_template(temp_analysis, data)

            form_analysis.append((docs_, analy_, plot_))

        return form_analysis
