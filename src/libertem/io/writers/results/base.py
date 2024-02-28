from libertem.common.analysis import AnalysisResultSet


class ResultFormatRegistry(type):
    """
    This is a metaclass to make ResultFormat subclasses register
    themselves. On import of the module the subclass is implemented in,
    the __init__ method is called, which in turn asks the class
    about their ID, a description, ...
    """

    # FIXME: TypedDict, needs typing-extensions
    registry: dict = {}

    def __init__(cls, name, args, bases):
        format_info = cls.format_info()
        if format_info is None:
            return
        cls.registry[format_info["id"]] = {
            "class": cls,
            "info": format_info,
        }

    @classmethod
    def get_available_formats(cls):
        return {
            identifier: {
                "identifier": identifier,
                "description": fmt['info']['description'],
            }
            for identifier, fmt in cls.registry.items()
        }

    @classmethod
    def get_format_by_id(cls, identifier) -> type['ResultFormat']:
        return cls.registry[identifier]['class']


class ResultFormat(metaclass=ResultFormatRegistry):
    @classmethod
    def format_info(cls):
        return None

    def __init__(self, result_set: AnalysisResultSet):
        self._result_set = result_set

    def serialize_to_buffer(self, buf):
        """
        Serialize to `buf`, which is a file-like, i.e. `io.BytesIO`
        """
        raise NotImplementedError()

    def get_result_keys(self):
        for k in self._result_set.keys():
            if not self._result_set[k].include_in_download:
                continue
            yield k

    def get_content_type(self):
        raise NotImplementedError()

    def get_filename(self):
        raise NotImplementedError()
