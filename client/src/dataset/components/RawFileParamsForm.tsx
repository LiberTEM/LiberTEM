import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoRAW, DatasetParamsRaw, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import Reshape from "./Reshape";

// some fields have different types in the form vs. in messages
type DatasetParamsRawForForm = Omit<DatasetParamsRaw,
    "type"
    | "path"
    | "nav_shape"
    | "sig_shape"> & {
        nav_shape: string,
        sig_shape: string,
    };

type MergedProps = FormikProps<DatasetParamsRawForForm> & OpenFormProps<DatasetParamsRaw, DatasetInfoRAW>;

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
    setFieldValue,
}) => {

    return (

        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="id_name">Name:</label>
                <ErrorMessage name="name" />
                <Field name="name" id="id_name" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_dtype">Datatype (uint16, uint32, float32, float64, &gt;u2, ..., can be anything that is <a href="https://numpy.org/doc/stable/reference/arrays.dtypes.html">understood by numpy as a dtype</a>):</label>
                <ErrorMessage name="dtype" />
                <Field name="dtype" id="id_dtype" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_enable_direct">Enable Direct I/O (for usage with fast SSDs and files much larger than RAM):</label>
                <ErrorMessage name="enable_direct" />
                <Field type="checkbox" name="enable_direct" checked={values.enable_direct} id="id_enable_direct" />
            </Form.Field>
            <Reshape navShape={values.nav_shape} sigShape={values.sig_shape} syncOffset={values.sync_offset} hideInfo={true} setFieldValue={setFieldValue} />
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsRaw, DatasetParamsRawForForm, DatasetInfoRAW>({
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        enable_direct: getInitial("enable_direct", false, initial),
        dtype: getInitial("dtype", "float32", initial),
        nav_shape: getInitial("nav_shape", "", initial).toString(),
        sig_shape: getInitial("sig_shape", "", initial).toString(),
        sync_offset: getInitial("sync_offset", 0, initial),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.RAW,
            name: values.name,
            dtype: values.dtype,
            enable_direct: values.enable_direct,
            nav_shape: parseNumList(values.nav_shape),
            sig_shape: parseNumList(values.sig_shape),
            sync_offset: values.sync_offset,
        }
    },
    type: DatasetTypes.RAW,
})(RawFileParamsForm);
