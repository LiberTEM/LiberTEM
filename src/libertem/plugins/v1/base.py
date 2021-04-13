import traitlets


class Plugin:
    udf_class = None  # either set `udf_class` or override `get_udf`
    schema_class = None  # optionally define a schema for parameters
    channels = None  # select channels for visualization, default all

    def __init__(self, input_params=None):
        self.input_params = input_params

    def get_udf_class(self):
        if self.udf_class is None:
            raise ValueError("please either set `udf_class` or override `get_udf_class`")
        return self.udf_class

    def get_udf(self):
        udf_class = self.get_udf_class()
        return udf_class(**self.get_params())

    def get_params(self):
        if self.schema_class is None:
            return {}
        try:
            params = self.schema_class(**self.input_params)
        except traitlets.TraitError:
            raise  # TODO: propagate validation error as our own exception type
        return params
