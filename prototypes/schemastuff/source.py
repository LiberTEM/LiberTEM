import json
from marshmallow import Schema, fields
from marshmallow_jsonschema import JSONSchema


class StartJobDetailsSchema(Schema):
    id = fields.String()
    dataset = fields.String()


class StartJobResponseSchema(Schema):
    status = fields.String()  # TODO: constant "ok"?
    messageType = fields.String()  # TODO: constant "START_JOB"
    job = fields.String()
    details = fields.Nested(StartJobDetailsSchema)


start_job_schema = StartJobResponseSchema()
json_schema = JSONSchema()
print(json.dumps(json_schema.dump(start_job_schema).data))
