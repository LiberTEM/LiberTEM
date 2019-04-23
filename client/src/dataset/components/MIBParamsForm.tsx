import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsMIB, DatasetTypes } from "../../messages";
import { getInitial, parseNumList } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsMIBForForm = Omit<DatasetParamsMIB,
    "path"
    | "type"
    | "tileshape"
    | "scan_size"
> & {
    tileshape: string,
    scan_size: string,
};

type FormValues = DatasetParamsMIBForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsMIB>;

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
                <label htmlFor="id_tileshape">Tileshape:</label>
                <input type="text" name="tileshape" id="id_tileshape" value={values.tileshape}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size">Scan Size:</label>
                <input type="text" name="scan_size" id="id_scan_size" value={values.scan_size}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsMIB>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        tileshape: getInitial("tileshape", "1, 8, 256, 256", initial).toString(),
        scan_size: getInitial("scan_size", "256, 256", initial).toString(),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.MIB,
            name: values.name,
            tileshape: parseNumList(values.tileshape),
            scan_size: parseNumList(values.scan_size),
        });
    }
})(RawFileParamsForm);
