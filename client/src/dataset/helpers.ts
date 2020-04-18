import { withFormik } from "formik";
import * as pathfind from 'path';
import { AdditionalInfo, DatasetTypes } from "../messages";
import { OpenFormProps } from "./types";
import { validateOpen } from "./validate";

export function parseNumList(nums: string) {
    return nums.split(",").filter(part => part.trim() !== "").map(part => +part);
}

export function getInitial<T extends object, K extends keyof T, V>(key: K, otherwise: V, values?: T): V | T[K] {
    if (!values) {
        return otherwise;
    }
    const res = values[key] !== undefined ? values[key] : otherwise;
    return res;
}

export function getInitialName<T extends object, K extends keyof T>(key: K, otherwise: string, values?: T): string | T[K] {
    if (!values) {
        return pathfind.basename(otherwise);
    }
    const res = values[key] !== undefined ? values[key] : pathfind.basename(otherwise);
    return res;
}

export function isKnownDatasetType(detectedType: string) {
  return (Object.keys(DatasetTypes).some((v) => v === detectedType) ? true : false);
}

export function isAdditionalInfo(param: string) {
  return (Object.keys(AdditionalInfo).some((v) => v === param) ? true : false);
}

export function hasKey<O>(obj: O, key: keyof any): key is keyof O {
  return key in obj
}

type FormToJsonFn<SubmitParams, FormParams> = (inParams: FormParams, path: string) => SubmitParams;
type PropsToValuesFn<SubmitParams, FormParams> = (props: OpenFormProps<SubmitParams>) => FormParams;

interface WithValidationOpts<SubmitParams extends object, FormParams> {
    formToJson: FormToJsonFn<SubmitParams, FormParams>,
    mapPropsToValues: PropsToValuesFn<SubmitParams, FormParams>,
    type: DatasetTypes,
    // WrappedComponent: React.FunctionComponent<FormikProps<FormParams> & OpenFormProps<SubmitParams>>
}

export function withValidation<SubmitParams extends object, FormParams>(
    opts: WithValidationOpts<SubmitParams, FormParams>
) {
    return withFormik<OpenFormProps<SubmitParams>, FormParams>({
        mapPropsToValues: opts.mapPropsToValues,
        handleSubmit: (values, formikBag) => {
            const { onSubmit, path } = formikBag.props;
            const submitData = opts.formToJson(values, path);
            onSubmit(submitData);
            formikBag.setSubmitting(false);
        },
        validate: (values, props) => {
            return validateOpen(opts.type, opts.formToJson(values, props.path));
        },
        enableReinitialize: true,
        validateOnChange: true,
        validateOnBlur: true,
    });
}
