import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoDM, DatasetParamsDM, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, adjustShapeWithBounds, parseShapeInCommaSeparatedString, validateSyncOffsetAndSigShape, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import BackendSelectionDropdown from "./BackendSelectionDropdown";
import Reshape from "./Reshape";

// some fields have different types in the form vs. in messages
type DatasetParamsDMForForm = Omit<DatasetParamsDM,
    "type"
    | "path"
    | "nav_shape"
    | "sig_shape"> & {
        nav_shape: string,
        sig_shape: string,
};

type FormValues = DatasetParamsDMForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsDM, DatasetInfoDM>;

const DMFileParamsForm: React.FC<MergedProps> = ({
    values,
    info,
    isSubmitting,
    handleSubmit,
    handleReset,
    isValidating,
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
            <label htmlFor="id_io_backend">I/O Backend:</label>
            <ErrorMessage name="io_backend" />
            <BackendSelectionDropdown
                value={values.io_backend}
                datasetTypeInfo={datasetTypeInfo}
                setFieldValue={setFieldValue} />
        </Form.Field>
        <Form.Field>
            <label htmlFor="id_force_c_order">Force C-order:{" "}
            <Field name="force_c_order" id="id_force_c_order" type="checkbox"/>
            </label>
        </Form.Field>
        <Reshape navShape={values.nav_shape} sigShape={values.sig_shape} syncOffset={values.sync_offset} imageCount={info?.image_count} setFieldValue={setFieldValue} setFieldTouched={setFieldTouched} />
        <Button primary type="submit" disabled={isSubmitting || isValidating}>Load Dataset</Button>
        <Button type="button" onClick={onCancel}>Cancel</Button>
        <Button type="button" onClick={handleReset}>Reset</Button>
    </Form>
)

export default withValidation<DatasetParamsDM, DatasetParamsDMForForm, DatasetInfoDM>({
    formToJson: (values, path) => ({
        path,
        type: DatasetTypes.DM,
        name: values.name,
        nav_shape: parseShapeInCommaSeparatedString(values.nav_shape),
        sig_shape: parseShapeInCommaSeparatedString(values.sig_shape),
        sync_offset: values.sync_offset,
        io_backend: values.io_backend,
        force_c_order: values.force_c_order,
    }),
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        nav_shape: adjustShapeWithBounds(getInitial("nav_shape", "", initial).toString(), "nav"),
        sig_shape: adjustShapeWithBounds(getInitial("sig_shape", "", initial).toString(), "sig"),
        sync_offset: getInitial("sync_offset", 0, initial),
        io_backend: getInitial("io_backend", undefined, initial),
        force_c_order: getInitial("force_c_order", false, initial),
    }),
    customValidation: (values, { info }) => validateSyncOffsetAndSigShape(
        info?.native_sig_shape,
        values.sig_shape,
        values.sync_offset,
        info?.image_count
    ),
    type: DatasetTypes.DM,
})(DMFileParamsForm);
