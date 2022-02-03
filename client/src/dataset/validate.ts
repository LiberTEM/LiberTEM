import Ajv, { ErrorObject } from 'ajv';
import { FormikErrors, FormikValues } from 'formik';
import { JsonSchema } from '../messages';

export const convertErrors = (errors: ErrorObject[]): FormikErrors<FormikValues> => {
    const res: FormikErrors<FormikValues> = {};
    errors.forEach(err => {
        const errorParts = err.instancePath.split('/');
        // flatten field names, convert from array to object
        // field_name[6] -> field_name
        // FIXME: doesn't support multiple errors per field yet (formik limitation?)
        const expr = /^\.([^[]+)(\[[^]+\])?$/
        const cleanedErrParts = errorParts
            .map((ep) => ep.replace(expr, (match, plainName) => plainName as string))
            .filter((ep) => ep.length > 0 && !/^[0-9]$/.test(ep));  // remove empty and numbers

        const fieldName = cleanedErrParts[0];  // this should now be the field name!
        res[fieldName] = err.message;
    });
    return res;
}

export const throwErrors = (validateErrors : ErrorObject[] | null = [], customValidateErrors: FormikErrors<FormikValues> = {}): FormikErrors<FormikValues> => {
    if (validateErrors || customValidateErrors) {
        const converted = validateErrors ? { ...convertErrors(validateErrors), ...customValidateErrors } : customValidateErrors;
        return converted;
    } else {
        throw new Error("unspecified error while validating fields");
    }
}

export const validateOpen = <T>(schema: JsonSchema, data: T, customValidateErrors?: FormikErrors<FormikValues>): FormikErrors<FormikValues> => {
    const ajv = new Ajv();
    const validate = ajv.compile(schema);
    const valid = validate(data);
    if (!valid || customValidateErrors) {
        const errs = throwErrors(validate.errors, customValidateErrors);
        return errs;
    }
    return {};
}
