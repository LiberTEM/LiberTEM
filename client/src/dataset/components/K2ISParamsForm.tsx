import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsK2IS, DatasetTypes } from "../../messages";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsK2ISForForm = Omit<DatasetParamsK2IS,
    "path"
    | "skipFrames"
    | "scanSize"
    | "type"> & {
    scanSize: string,
    skipFrames: number,
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
                <label htmlFor="scanSize">Scan Size:</label>
                <input type="text" name="scanSize" value={values.scanSize}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="scanSize">Skip Frames:</label>
                <input type="text" name="skipFrames" value={values.skipFrames}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>

            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
        </Form>
    )
}

function parseNumList(nums: string) {
    return nums.split(",").map(part => +part);
}

export default withFormik<OpenFormProps<DatasetParamsK2IS>, FormValues>({
    mapPropsToValues: () => ({
        name: "",
        scanSize: "32, 32",
        skipFrames: 0,
        dtype: "float32",
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.K2IS,
            name: values.name,
            skipFrames: +values.skipFrames,
            scanSize: parseNumList(values.scanSize),
        });
    }
})(K2ISFileParamsForm);

