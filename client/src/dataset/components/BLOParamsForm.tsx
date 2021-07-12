import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoBLO, DatasetParamsBLO, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, validateSyncOffsetAndSigShape, withValidation } from "../helpers";
import { OpenFormProps } from "../types";
import Reshape from "./Reshape";

// some fields have different types in the form vs. in messages
type DatasetParamsBLOForForm = Omit<DatasetParamsBLO,
    "type"
    | "path"
    | "nav_shape"
    | "sig_shape"> & {
        nav_shape: string,
        sig_shape: string,
};

type MergedProps = FormikProps<DatasetParamsBLOForForm> & OpenFormProps<DatasetParamsBLO, DatasetInfoBLO>;

const BLOFileParamsForm: React.FC<MergedProps> = ({
    values,
    info,
    isSubmitting,
    handleSubmit,
    handleReset,
    isValidating,
    onCancel,
    setFieldValue,
}) => (
    <Form onSubmit={handleSubmit}>
        <Form.Field>
            <label htmlFor="id_name">Name:</label>
            <ErrorMessage name="name" />
            <Field name="name" id="id_name" />
        </Form.Field>
        <Reshape navShape={values.nav_shape} sigShape={values.sig_shape} syncOffset={values.sync_offset} imageCount={info?.image_count} setFieldValue={setFieldValue} />
        <Button primary type="submit" disabled={isSubmitting || isValidating}>Load Dataset</Button>
        <Button type="button" onClick={onCancel}>Cancel</Button>
        <Button type="button" onClick={handleReset}>Reset</Button>
    </Form>
)

export default withValidation<DatasetParamsBLO, DatasetParamsBLOForForm, DatasetInfoBLO>({
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        nav_shape: getInitial("nav_shape", "", initial).toString(),
        sig_shape: getInitial("sig_shape", "", initial).toString(),
        sync_offset: getInitial("sync_offset", 0, initial),
    }),
    formToJson: (values, path) => ({
        path,
        type: DatasetTypes.BLO,
        name: values.name,
        nav_shape: parseNumList(values.nav_shape),
        sig_shape: parseNumList(values.sig_shape),
        sync_offset: values.sync_offset,
    }),
    customValidation: (values, { info }) =>
        validateSyncOffsetAndSigShape(
            info?.native_sig_shape,
            values.sig_shape,
            values.sync_offset,
            info?.image_count
        ),
    type: DatasetTypes.BLO,
})(BLOFileParamsForm);
