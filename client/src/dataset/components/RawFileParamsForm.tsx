import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsRaw, DatasetTypes } from "../../messages";
import { getInitial, parseNumList } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsRawForForm = Omit<DatasetParamsRaw,
    "type"
    | "tileshape"
    | "path"
    | "scan_size"
    | "detector_size_raw"
    | "crop_detector_to"> & {
        tileshape: string,
        scan_size: string
        detector_size_raw: string,
        crop_detector_to: string,
    };

type FormValues = DatasetParamsRawForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsRaw>;

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
                <label htmlFor="name">Name:</label>
                <input type="text" name="name" value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.name && touched.name && errors.name}
            </Form.Field>
            <Form.Field>
                <label htmlFor="tileshape">Tileshape:</label>
                <input type="text" name="tileshape" value={values.tileshape}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="scan_size">Scan Size:</label>
                <input type="text" name="scan_size" value={values.scan_size}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="dtype">Datatype (uint16, uint32, float32, float64, &gt;u2, ..., can be anything that is <a href="https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.dtypes.html">understood by numpy as a dtype</a>):</label>
                <input type="text" name="dtype" value={values.dtype}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>

            <Form.Field>
                <label htmlFor="detector_size_raw">Detector Size (as in the file):</label>
                <input type="text" name="detector_size_raw" value={values.detector_size_raw}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="crop_detector_to">Detector Size Crop:</label>
                <input type="text" name="crop_detector_to" value={values.crop_detector_to}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsRaw>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        tileshape: getInitial("tileshape", "1, 8, 128, 128", initial),
        detector_size_raw: getInitial("detector_size_raw", "130, 128", initial),
        crop_detector_to: getInitial("crop_detector_to", "128, 128", initial),
        scan_size: getInitial("scan_size", "256, 256", initial),
        dtype: getInitial("dtype", "float32", initial),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.RAW,
            name: values.name,
            dtype: values.dtype,
            tileshape: parseNumList(values.tileshape),
            scan_size: parseNumList(values.scan_size),
            detector_size_raw: parseNumList(values.detector_size_raw),
            crop_detector_to: parseNumList(values.crop_detector_to),
        });
    }
})(RawFileParamsForm);
