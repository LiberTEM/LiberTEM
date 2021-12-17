from typing import Optional
from libertem.web.notebook_generator.template import TemplateBase


class GeneratorHelper(TemplateBase):
    """
    Helper class for generating code corresponding to analysis.

    code for creating and running analysis, saving the result,
    plotting the result, documentation, specific dependencies
    are generated.
    """

    short_name: Optional[str] = None
    api: Optional[str] = None

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

    def get_save(self):
        '''
        Get code for saving result.
        '''
        data = {'short': self.short_name}
        save = self.format_template(self.temp_save, data)
        return save

    def format_docs(self, title, docs_rst):
        """
        function to format docs for notebook
        """
        docs = f'# {title}\n\n<pre>{docs_rst}</pre>'
        return docs

    def get_roi(self):
        """
        Get code for roi
        """
        roi = ["nx, ny = ds.shape.nav"]
        params = self.params['roi']
        if params == {}:
            roi = ["roi = None"]

        elif params["shape"] == "disk":
            x = params["cx"]
            y = params["cy"]
            r = params["r"]
            p = f"centerX={x}, centerY={y}, imageSizeX=nx, imageSizeY=ny, radius={r}"
            roi.append(f"roi = masks.circular({p})")

        elif params["shape"] == "rect":
            x = params['x']
            y = params['y']
            width = params['width']
            height = params['height']
            p = f"X={x}, Y={y}, Width={width}, Height={height}, imageSizeX=nx, imageSizeY=ny"
            roi.append(f"roi = masks.rectangular({p})")

        return '\n'.join(roi)

    def get_analysis(self):
        '''
        get code corresponding to create and run analysis.
        override for adding specific code for analysis
        subclasses.
        '''
        params_ = self.convert_params()

        data = {
            'short': self.short_name,
            'analysis_api': self.api,
            'params': params_,
        }

        analy_ = self.format_template(self.temp_analysis, data)

        return analy_
