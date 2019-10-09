import { DatasetParamsSER } from "../../messages";

import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetTypes } from "../../messages";
import { getInitial } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsSERForForm = Omit<DatasetParamsSER,
    "path" | "type">;

type FormValues = DatasetParamsSERForForm


type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsSER>;
const SERParamsForm: React.SFC<MergedProps> = ({
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
                <input type="text" name="name" id="id_name" value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.name && touched.name && errors.name}
            </Form.Field>

            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsSER>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.SER,
            name: values.name,
        });
    },
    enableReinitialize: true,
})(SERParamsForm);