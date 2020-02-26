import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsFRMS6, DatasetTypes } from "../../messages";
import { getInitialName, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsFRMS6ForForm = Omit<DatasetParamsFRMS6,
    "path" | "type">;

type MergedProps = FormikProps<DatasetParamsFRMS6ForForm> & OpenFormProps<DatasetParamsFRMS6>;
const FRMS6ParamsForm: React.SFC<MergedProps> = ({
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

            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsFRMS6, DatasetParamsFRMS6ForForm>({
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name",path,initial),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.FRMS6,
            name: values.name,
        };
    },
    type: DatasetTypes.FRMS6,
})(FRMS6ParamsForm);
