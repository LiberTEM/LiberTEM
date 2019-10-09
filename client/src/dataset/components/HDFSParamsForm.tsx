import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsHDFS, DatasetTypes } from "../../messages";
import { getInitial, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

type DatasetParamsHDFForForm = Omit<DatasetParamsHDFS, "path" | "type" | "tileshape"> & {
    tileshape: string,
};

type MergedProps = FormikProps<DatasetParamsHDFForForm> & OpenFormProps<DatasetParamsHDFS>;

const HDFSParamsForm: React.SFC<MergedProps> = ({
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
                <label htmlFor="id_tileshape">Tileshape:</label>
                <ErrorMessage name="tileshape" />
                <Field name="tileshape" id="id_tileshape" />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsHDFS, DatasetParamsHDFForForm>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        tileshape: getInitial("tileshape", "1, 8, 128, 128", initial).toString(),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.HDFS,
            name: values.name,
            tileshape: parseNumList(values.tileshape),
        };
    },
    type: DatasetTypes.HDFS,
})(HDFSParamsForm);