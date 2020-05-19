import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsMIB, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import TupleInput from "./TupleInput";

// some fields have different types in the form vs. in messages
type DatasetParamsMIBForForm = Omit<DatasetParamsMIB,
    "path"
    | "type"
    | "scan_size"
> & {
    scan_size: string,
};

type FormValues = DatasetParamsMIBForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsMIB>;

const MIBFileParamsForm: React.SFC<MergedProps> = ({
    values,
    touched,
    errors,
    dirty,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    handleReset,
    isValidating,
    onCancel,
    setFieldValue,
}) => {

    return (
        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="id_name">Name:</label>
                <ErrorMessage name="name" />
                <Field name="name" id="id_name" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size_0">Scan Size:</label>
                <ErrorMessage name="scan_size" />
                <TupleInput value={values.scan_size} minLen={2} maxLen={2} fieldName="scan_size" setFieldValue={setFieldValue} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting || isValidating}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsMIB, DatasetParamsMIBForForm>({
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.MIB,
            name: values.name,
            scan_size: parseNumList(values.scan_size),
        }
    },
    mapPropsToValues: ({path, initial }) => ({
        name: getInitialName("name",path,initial),
        scan_size: getInitial("scan_size", "", initial).toString(),
    }),
    type: DatasetTypes.MIB,
})(MIBFileParamsForm);
