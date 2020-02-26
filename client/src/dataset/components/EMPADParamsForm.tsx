import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsEMPAD, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsEMPADForForm = Omit<DatasetParamsEMPAD,
    "path"
    | "type"
    | "scan_size"
> & {
    scan_size: string,
};

type MergedProps = FormikProps<DatasetParamsEMPADForForm> & OpenFormProps<DatasetParamsEMPAD>;

const EMPADParamsForm: React.SFC<MergedProps> = ({
    values,
    touched,
    errors,
    dirty,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    handleReset,
    onCancel,
}) => {
    return (
        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="id_name">Name:</label>
                <ErrorMessage name="name" />
                <Field name="name" id="id_name" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size">Scan Size:</label>
                <ErrorMessage name="scan_size" />
                <Field name="scan_size" id="id_scan_size" />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsEMPAD, DatasetParamsEMPADForForm>({
    mapPropsToValues: ({path, initial }) => ({
        name: getInitialName("name",path,initial),
        scan_size: getInitial("scan_size", "", initial).toString(),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.EMPAD,
            name: values.name,
            scan_size: parseNumList(values.scan_size),
        };
    },
    type: DatasetTypes.EMPAD,
})(EMPADParamsForm);
