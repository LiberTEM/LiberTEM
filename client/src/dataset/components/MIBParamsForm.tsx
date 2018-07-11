import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsMIB, DatasetTypes } from "../../messages";

// some fields have different types in the form vs. in messages
type DatasetParamsMIBForForm = Omit<DatasetParamsMIB,
    "type"
    | "tileshape"
    | "scanSize"
    > & {
    tileshape: string,
    scanSize: string
};

type FormValues = DatasetParamsMIBForForm

interface FormProps {
    onSubmit: (params: DatasetParamsMIB) => void
    onCancel: () => void,
}

type MergedProps = FormikProps<FormValues> & FormProps;

const RawFileParamsForm: React.SFC<MergedProps> = ({
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
                <label htmlFor="filesPattern">File Pattern:</label>
                <input type="text" name="filesPattern" value={values.filesPattern}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="tileshape">Tileshape:</label>
                <input type="text" name="tileshape" value={values.tileshape}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="scanSize">Scan Size:</label>
                <input type="text" name="scanSize" value={values.scanSize}
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

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: () => ({
        name: "",
        tileshape: "1, 8, 256, 256",
        scanSize: "256, 256",
        filesPattern: "",
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: DatasetTypes.MIB,
            name: values.name,
            filesPattern: values.filesPattern,
            tileshape: parseNumList(values.tileshape),
            scanSize: parseNumList(values.scanSize),
        });
    }
})(RawFileParamsForm);
