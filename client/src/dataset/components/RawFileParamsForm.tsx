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
    | "path"
    | "tileshape"
    | "scan_size"
    | "detector_size"> & {
        tileshape: string,
        scan_size: string
        detector_size: string,
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
                <label htmlFor="id_name">Name:</label>
                <input type="text" name="name" id="id_name" value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.name && touched.name && errors.name}
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_tileshape">Tileshape:</label>
                <input type="text" name="tileshape" id="id_tileshape" value={values.tileshape}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_scan_size">Scan Size:</label>
                <input type="text" name="scan_size" id="id_scan_size" value={values.scan_size}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_dtype">Datatype (uint16, uint32, float32, float64, &gt;u2, ..., can be anything that is <a href="https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.dtypes.html">understood by numpy as a dtype</a>):</label>
                <input type="text" name="dtype" id="id_dtype" value={values.dtype}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>

            <Form.Field>
                <label htmlFor="id_detector_size">Detector Size (as in the file):</label>
                <input type="text" name="detector_size" id="id_detector_size" value={values.detector_size}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_enable_direct">Enable Direct I/O (for usage with fast SSDs and files much larger than RAM):</label>
                <input type="checkbox" name="enable_direct" id="id_enable_direct" checked={values.enable_direct}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button type="button" onClick={onCancel}>Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withFormik<OpenFormProps<DatasetParamsRaw>, FormValues>({
    mapPropsToValues: ({ initial }) => ({
        name: getInitial("name", "", initial),
        enable_direct: getInitial("enable_direct", false, initial),
        tileshape: getInitial("tileshape", "1, 8, 128, 128", initial).toString(),
        detector_size: getInitial("detector_size", "", initial).toString(),
        scan_size: getInitial("scan_size", "", initial).toString(),
        dtype: getInitial("dtype", "float32", initial),
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        onSubmit({
            path,
            type: DatasetTypes.RAW,
            name: values.name,
            dtype: values.dtype,
            enable_direct: values.enable_direct,
            tileshape: parseNumList(values.tileshape),
            scan_size: parseNumList(values.scan_size),
            detector_size: parseNumList(values.detector_size),
        });
    },
    enableReinitialize: true,
})(RawFileParamsForm);
