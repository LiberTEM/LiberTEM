import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsEMPAD, DatasetTypes } from "../../messages";
import { getInitial, parseNumList } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsEMPADForForm = Omit<DatasetParamsEMPAD,
    "path"
    | "type"
    | "scan_size"
> & {
    scan_size: string,
};

type FormValues = DatasetParamsEMPADForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsEMPAD>;

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
                <label htmlFor="id_name">Name:</label>
                <input type="text" name="name" id="id_name" value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.name && touched.name && errors.name}
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size">Scan Size:</label>
                <input type="text" name="scan_size" id="id_scan_size" value={values.scan_size}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsEMPAD>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        scan_size: getInitial("scan_size", "", initial).toString(),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.EMPAD,
            name: values.name,
            scan_size: parseNumList(values.scan_size),
        });
    },
    enableReinitialize: true,
})(RawFileParamsForm);

