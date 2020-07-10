from string import Template


class TemplateBase:
    '''
    Base class for template strings.
    '''

    temp_ds = ['params = $params',
               'ds = ctx.load("$type", **params)']

    temp_dep = ["import matplotlib.pyplot as plt",
                "import libertem.api as lt",
                "import numpy as np",
                "from libertem.analysis.getroi import get_roi",
                "import numpy as np"]

    temp_conn = ['cluster = executor.dask.DaskJobExecutor.connect("$conn_url")',
                 "ctx = lt.Context(executor=cluster)"]

    temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                     "$roi",
                     "udf = ${short}_analysis.get_udf()",
                     "${short}_result = ctx.run_udf(ds, udf, roi, progress=True)"]

    temp_roi = ["roi_params = $roi_params",
                "roi = get_roi(roi_params, ds.shape.nav)"]

    temp_save = ["np.save('${short}_result.npy', ${short}_result['intensity'])"]

    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)
