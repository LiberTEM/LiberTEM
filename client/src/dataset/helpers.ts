import { withFormik } from "formik";
import { DatasetTypes } from "../messages";
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