from string import Template


class GeneratorHelper:

    short_name = None
    api = None

    def __init__(self, params):
        self.params = params

    # common in here and code_template
    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)

    def get_dependency():
        return None

    def convert_params(self):
        return None

    def get_plot(self):
        return None

    def get_docs(self):
        return None

    def get_roi_code(self):

        if 'roi' in self.params.keys():
            temp_roi = ["roi_params = $roi_params",
                        "roi = get_roi(roi_params, ds.shape.nav)"]
            data = {'roi_params': self.params['roi']}
            roi = self.format_template(temp_roi, data)
        else:
            roi = f"roi = {self.short_name}_analysis.get_roi()"

        return roi

    def get_analysis(self):

        temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                         "$roi",
                         "udf = ${short}_analysis.get_udf()",
                         "${short}_result = ctx.run_udf(ds, udf, roi, progress=True)"]

        params_ = self.convert_params()
        roi = self.get_roi_code()

        data = {'short': self.short_name,
                'analysis_api': self.api,
                'params': params_,
                'roi': roi}

        analy_ = self.format_template(temp_analysis, data)

        return analy_
