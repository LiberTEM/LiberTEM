from typing import Optional
import jsonschema


class MessageConverter:
    SCHEMA: Optional[dict] = None

    def validate(self, raw_data):
        if self.SCHEMA is None:
            raise NotImplementedError("please specify a SCHEMA")
        # FIXME: re-throw own exception type?
        jsonschema.validate(schema=self.SCHEMA, instance=raw_data)

    def to_python(self, raw_data):
        """
        validate and convert from JSONic data structures to python data structures
        """
        from libertem.io.dataset.base import IOBackend
        self.validate(raw_data)
        results = self.convert_to_python(raw_data)
        if "io_backend" in raw_data:
            results["io_backend"] = IOBackend.get_cls_by_id(raw_data["io_backend"])()
        return results

    def convert_to_python(self, raw_data):
        """
        override this method to provide your own conversion
        """
        return raw_data

    def from_python(self, raw_data):
        """
        convert from python data structures to JSONic data structures and validate
        """
        converted = self.convert_from_python(raw_data)
        self.validate(converted)
        return converted

    def convert_from_python(self, raw_data):
        """
        override this method to provide your own conversion
        """
        return raw_data
