import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsRaw, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsRawForForm = Omit<DatasetParamsRaw,
    "type"
    | "path"
    | "scan_size"
    | "detector_size"> & {
        scan_size: string
        detector_size: string,
    };

type MergedProps = FormikProps<DatasetParamsRawForForm> & OpenFormProps<DatasetParamsRaw>;

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
                <ErrorMessage name="name" />
                <Field name="name" id="id_name" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size">Scan Size:</label>
                <ErrorMessage name="scan_size" />
                <Field name="scan_size" id="id_scan_size" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_dtype">Datatype (uint16, uint32, float32, float64, &gt;u2, ..., can be anything that is <a href="https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.dtypes.html">understood by numpy as a dtype</a>):</label>
                <ErrorMessage name="dtype" />
                <Field name="dtype" id="id_dtype" />
            </Form.Field>

            <Form.Field>
                <label htmlFor="id_detector_size">Detector Size (pixels, example: 256,256):</label>
                <ErrorMessage name="detector_size" />
                <Field name="detector_size" id="id_detector_size" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_enable_direct">Enable Direct I/O (for usage with fast SSDs and files much larger than RAM):</label>
                <ErrorMessage name="enable_direct" />
                <Field type="checkbox" name="enable_direct" checked={values.enable_direct} id="id_enable_direct" />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsRaw, DatasetParamsRawForForm>({
    mapPropsToValues: ({path, initial }) => ({
        name: getInitialName("name",path,initial),
        enable_direct: getInitial("enable_direct", false, initial),
        detector_size: getInitial("detector_size", "", initial).toString(),
        scan_size: getInitial("scan_size", "", initial).toString(),
        dtype: getInitial("dtype", "float32", initial),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.RAW,
            name: values.name,
            dtype: values.dtype,
            enable_direct: values.enable_direct,
            scan_size: parseNumList(values.scan_size),
            detector_size: parseNumList(values.detector_size),
        }
    },
    type: DatasetTypes.RAW,
})(RawFileParamsForm);
