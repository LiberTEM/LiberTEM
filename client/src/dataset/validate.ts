import Ajv, { ErrorObject } from 'ajv';
import { FormikErrors, FormikValues } from 'formik';
import { DataSetOpenSchemaResponse } from '../messages';
import { getSchema } from './api';

export function convertErrors(errors: ErrorObject[]): FormikErrors<FormikValues> {
    const res: FormikErrors<FormikValues> = {};
    errors.forEach(err => {
        // flatten field names, convert from array to object
        // FIXME: doesn't support multiple errors per field yet (formik limitation?)
        const expr = /^\.([^[]+)(\[[^]+\])?$/
        const fieldName = err.dataPath.replace(expr, (match, plainName) => {
            return plainName;
        });
        res[fieldName] = err.message;
    });
    return res;
}

export async function validateOpen(type: string, data: object) {
    return getSchema(type).then((schemaResponse: DataSetOpenSchemaResponse) => {
        if (schemaResponse.status === "error") {
            throw new Error(schemaResponse.msg);
        }
        // FIXME: cache compiled schema
        const schema = schemaResponse.schema;
        const ajv = new Ajv();
        const validate = ajv.compile(schema);
        const valid = validate(data);
        if (!valid) {
            if (validate.errors) {
                const converted = convertErrors(validate.errors);
                throw converted;
            } else {
                throw new Error("unspecified error while validating fields");
            }
        }
    })
}