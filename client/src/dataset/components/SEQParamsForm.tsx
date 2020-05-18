import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsSEQ, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import ScanSize from "./ScanSize";

// some fields have different types in the form vs. in messages
type DatasetParamsSEQForForm = Omit<DatasetParamsSEQ,
    "path"
    | "type"
    | "scan_size"
> & {
    scan_size: string,
};

type FormValues = DatasetParamsSEQForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsSEQ>;

const SEQFileParamsForm: React.SFC<MergedProps> = ({
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
                <ScanSize value={values.scan_size} minScan={2} maxScan={4} setFieldValue={setFieldValue} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting || isValidating}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsSEQ, DatasetParamsSEQForForm>({
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.SEQ,
            name: values.name,
            scan_size: parseNumList(values.scan_size),
        }
    },
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        scan_size: getInitial("scan_size", "", initial).toString(),
    }),
    type: DatasetTypes.SEQ,
})(SEQFileParamsForm);
