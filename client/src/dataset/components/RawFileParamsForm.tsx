import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoRAW, DatasetParamsRaw, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, adjustShapeWithBounds, parseShapeInCommaSeparatedString, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import BackendSelectionDropdown from "./BackendSelectionDropdown";
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

const RawFileParamsForm: React.FC<MergedProps> = ({
    values,
    isSubmitting,
    handleSubmit,
    handleReset,
    onCancel,
    setFieldValue,
    setFieldTouched,
    datasetTypeInfo,
}) => (
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
            <label htmlFor="id_io_backend">I/O Backend:</label>
            <ErrorMessage name="io_backend" />
            <BackendSelectionDropdown
                value={values.io_backend}
                datasetTypeInfo={datasetTypeInfo}
                setFieldValue={setFieldValue} />
        </Form.Field>
        <Reshape navShape={values.nav_shape} sigShape={values.sig_shape} syncOffset={values.sync_offset} hideInfo setFieldValue={setFieldValue} setFieldTouched={setFieldTouched} />
        <Button primary type="submit" disabled={isSubmitting}>Load Dataset</Button>
        <Button type="button" onClick={onCancel}>Cancel</Button>
        <Button type="button" onClick={handleReset}>Reset</Button>
    </Form>
)

export default withValidation<DatasetParamsRaw, DatasetParamsRawForForm, DatasetInfoRAW>({
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        dtype: getInitial("dtype", "float32", initial),
        nav_shape: adjustShapeWithBounds(getInitial("nav_shape", "", initial).toString(), "nav"),
        sig_shape: adjustShapeWithBounds(getInitial("sig_shape", "", initial).toString(), "sig"),
        sync_offset: getInitial("sync_offset", 0, initial),
        io_backend: getInitial("io_backend", undefined, initial),
    }),
    formToJson: (values, path) => ({
        path,
        type: DatasetTypes.RAW,
        name: values.name,
        dtype: values.dtype,
        nav_shape: parseShapeInCommaSeparatedString(values.nav_shape),
        sig_shape: parseShapeInCommaSeparatedString(values.sig_shape),
        sync_offset: values.sync_offset,
        io_backend: values.io_backend,
    }),
    type: DatasetTypes.RAW,
})(RawFileParamsForm);
