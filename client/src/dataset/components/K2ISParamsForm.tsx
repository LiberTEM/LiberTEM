import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsK2IS, DatasetTypes } from "../../messages";
import { getInitial, parseNumList } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsK2ISForForm = Omit<DatasetParamsK2IS,
    "path"
    | "scan_size"
    | "type"> & {
    scan_size: string,
};

type FormValues = DatasetParamsK2ISForForm


type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsK2IS>;

const K2ISFileParamsForm: React.SFC<MergedProps> = ({
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
                <label htmlFor="name">Name:</label>
                <input type="text" name="name" value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.name && touched.name && errors.name}
            </Form.Field>
            <Form.Field>
                <label htmlFor="scan_size">Scan Size:</label>
                <input type="text" name="scan_size" value={values.scan_size}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>

            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsK2IS>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        scan_size: getInitial("scan_size", "32, 32", initial),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.K2IS,
            name: values.name,
            scan_size: parseNumList(values.scan_size),
        });
    }
})(K2ISFileParamsForm);

