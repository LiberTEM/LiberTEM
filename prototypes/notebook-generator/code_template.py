from string import Template


class CodeTemplate():
    def __init__(self, connection, dataset, compound_analysis):
        self.conn = connection
        self.ds = dataset
        self.compound_analysis = compound_analysis

    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)

    def get_short(self, analysis):
        short_abb = {
            'SumAnalysis': 'sum',
            'RingMaskAnalysis': 'ring'
        }
        return short_abb[analysis]

    def dependency(self):
        temp_dep = ["import matplotlib.pyplot as plt",
                    "import libertem.api as lt",
                    "import numpy as np",
                    "from libertem.analysis import $analysis"]

        analysis = ", ".join(self.compound_analysis.keys())
        data = {'analysis': analysis}
        return self.format_template(temp_dep, data)

    # Any better names ?
    def modifiers(self):
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

    def dataset(self):
        ds_type = self.ds['type']
        ds_params = self.ds['params']
        temp_ds = ['params = $params',
                   'ds = ctx.load("$type", **params)']
        data = {'type': ds_type, 'params': ds_params}
        return self.format_template(temp_ds, data)

    def analysis(self):
        temp_analy = ["${short}_analysis = $analysis(dataset=ds, parameters=$parameters)",
                      "${short}_result = ctx.run(${short}_analysis, progress=True)"]
        temp_plot = ["plt.figure()",
                     "plt.imshow(${short}_result[0].visualized)"]
        form_analysis = []
        for analysis, parameters in self.compound_analysis.items():
            short = self.get_short(analysis)
            data = {'short': short, 'analysis': analysis, 'parameters': parameters}
            analy_ = self.format_template(temp_analy, data)
            title_ = f"# {analysis}"
            plot_ = self.format_template(temp_plot, {'short': short})
            form_analysis.append((title_, analy_, plot_))
        return form_analysis

    def plot(self):
        temp_plot = ["plt.figure()",
                     "plt.imshow(${short}_result[0].visualized)"]
        return '\n'.join(temp_plot)
