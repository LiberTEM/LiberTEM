from libertem.web.notebook_generator.template import TemplateBase


class GeneratorHelper(TemplateBase):
    """
    Helper class for generating code corresponding
    to analysis subclasses
    """

    short_name = None
    api = None

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        """
        Get analysis dependencies.
        """
        return None

    def convert_params(self):
        """
        Format analysis parameters.
        """
        return None

    def get_plot(self):
        """
        Get code for ploting analysis.
        """
        return None

    def get_docs(self):
        """
        Get documentation for analysis.
        """
        return None

    def format_docs(self, title, docs_rst):
        """
        function to format docs for notebook
        """
        docs = f'# {title}\n\n<pre>{docs_rst}</pre>'
        return docs

    def get_roi_code(self):
        '''
        Get code for roi
        '''
        if 'roi' in self.params.keys():
            data = {'roi_params': self.params['roi']}
            roi = self.format_template(self.temp_roi, data)
        else:
            roi = f"roi = {self.short_name}_analysis.get_roi()"

        return roi

    def get_analysis(self):
        '''
        get code corresponding to create and run analysis.
        override for adding specific code for analysis
        subclasses.
        '''
        params_ = self.convert_params()
        roi = self.get_roi_code()

        data = {'short': self.short_name,
                'analysis_api': self.api,
                'params': params_,
                'roi': roi}

        analy_ = self.format_template(self.temp_analysis, data)

        return analy_
