import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsMIB, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
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
}) => {
    return (
        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="id_name">Name:</label>
                <ErrorMessage name="name" />
                <Field name="name" id="id_name" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_tileshape">Tileshape:</label>
                <ErrorMessage name="tileshape" />
                <Field name="tileshape" id="id_tileshape" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size">Scan Size:</label>
                <ErrorMessage name="scan_size" />
                <Field name="scan_size" id="id_scan_size" />
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
            tileshape: parseNumList(values.tileshape),
            scan_size: parseNumList(values.scan_size),
        }
    },
    mapPropsToValues: ({path, initial }) => ({
        name: getInitialName("name",path,initial),
        tileshape: getInitial("tileshape", "1, 8, 256, 256", initial).toString(),
        scan_size: getInitial("scan_size", "", initial).toString(),
    }),
    type: DatasetTypes.MIB,
})(MIBFileParamsForm);
