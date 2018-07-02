import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { DatasetParamsHDFS, DatasetTypes } from "../../messages";

type Diff<T extends string | number | symbol, U extends string | number | symbol> = ({ [P in T]: P } & { [P in U]: never } & { [x: string]: never })[T];
type Omit<T, K extends keyof T> = Pick<T, Diff<keyof T, K>>;

type RawDatasetParamsHDFS = Omit<DatasetParamsHDFS, "type" | "tileshape"> & { tileshape: string };

type FormValues = RawDatasetParamsHDFS

interface FormProps {
    onSubmit: (params: DatasetParamsHDFS) => void
    onCancel: () => void,
}

type MergedProps = FormikProps<FormValues> & FormProps;

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
                <label htmlFor="name">Name:</label>
                <input type="text" name="name" value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.name && touched.name && errors.name}
            </Form.Field>
            <Form.Field>
                <label htmlFor="path">Path:</label>
                <input type="text" name="path" value={values.path}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="tileshape">Tileshape:</label>
                <input type="text" name="tileshape" value={values.tileshape}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
        </Form>
    )
}

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: () => ({
        name: "",
        tileshape: "1, 8, 128, 128",
        path: "",
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: DatasetTypes.HDFS,
            name: values.name,
            path: values.path,
            tileshape: values.tileshape.split(",").map(part => +part),
        });
    }
})(HDFSParamsForm);