import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsHDF5, DatasetTypes } from "../../messages";
import { getInitial } from "../helpers";
import { OpenFormProps } from "../types";

type DatasetParamsHDF5ForForm = Omit<DatasetParamsHDF5, "path" | "type" | "tileshape"> & { tileshape: string, };

type FormValues = DatasetParamsHDF5ForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsHDF5>;

const HDF5ParamsForm: React.SFC<MergedProps> = ({
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
                <label htmlFor="id_ds_path">HDF5 Dataset Path:</label>
                <input type="text" name="ds_path" id="id_ds_path" value={values.ds_path}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_tileshape">Tileshape:</label>
                <input type="text" name="tileshape" id="id_tileshape" value={values.tileshape}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button onClick={onCancel} >Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsHDF5>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        tileshape: getInitial("tileshape", "1, 8, 128, 128", initial).toString(),
        ds_path: getInitial("ds_path", "", initial),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.HDF5,
            name: values.name,
            ds_path: values.ds_path,
            tileshape: values.tileshape.split(",").map(part => +part),
        });
    },
    enableReinitialize: true,
})(HDF5ParamsForm);