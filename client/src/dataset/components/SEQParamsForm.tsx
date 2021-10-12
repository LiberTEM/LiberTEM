import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoSEQ, DatasetParamsSEQ, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, validateSyncOffsetAndSigShape, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import BackendSelectionDropdown from "./BackendSelectionDropdown";
import Reshape from "./Reshape";

// some fields have different types in the form vs. in messages
type DatasetParamsSEQForForm = Omit<DatasetParamsSEQ,
    "type"
    | "path"
    | "nav_shape"
    | "sig_shape"> & {
        nav_shape: string,
        sig_shape: string,
};

type FormValues = DatasetParamsSEQForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsSEQ, DatasetInfoSEQ>;

const SEQFileParamsForm: React.FC<MergedProps> = ({
    values,
    info,
    isSubmitting,
    handleSubmit,
    handleReset,
    isValidating,
    onCancel,
    setFieldValue,
    datasetTypeInfo,
}) => (
    <Form onSubmit={handleSubmit}>
        <Form.Field>
            <label htmlFor="id_name">Name:</label>
            <ErrorMessage name="name" />
            <Field name="name" id="id_name" />
        </Form.Field>
        <Reshape navShape={values.nav_shape} sigShape={values.sig_shape} syncOffset={values.sync_offset} imageCount={info?.image_count} setFieldValue={setFieldValue} />
        <Form.Field>
            <label htmlFor="id_io_backend">I/O Backend:</label>
            <ErrorMessage name="io_backend" />
            <BackendSelectionDropdown
                value={values.io_backend}
                datasetTypeInfo={datasetTypeInfo}
                setFieldValue={setFieldValue} />
        </Form.Field>
        <Button primary type="submit" disabled={isSubmitting || isValidating}>Load Dataset</Button>
        <Button type="button" onClick={onCancel}>Cancel</Button>
        <Button type="button" onClick={handleReset}>Reset</Button>
    </Form>
)

export default withValidation<DatasetParamsSEQ, DatasetParamsSEQForForm, DatasetInfoSEQ>({
    formToJson: (values, path) => ({
        path,
        type: DatasetTypes.SEQ,
        name: values.name,
        nav_shape: parseNumList(values.nav_shape),
        sig_shape: parseNumList(values.sig_shape),
        sync_offset: values.sync_offset,
        io_backend: values.io_backend,
    }),
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        nav_shape: getInitial("nav_shape", "", initial).toString(),
        sig_shape: getInitial("sig_shape", "", initial).toString(),
        sync_offset: getInitial("sync_offset", 0, initial),
        io_backend: getInitial("io_backend", undefined, initial),
    }),
    customValidation: (values, { info }) => validateSyncOffsetAndSigShape(
        info?.native_sig_shape,
        values.sig_shape,
        values.sync_offset,
        info?.image_count
    ),
    type: DatasetTypes.SEQ,
})(SEQFileParamsForm);
