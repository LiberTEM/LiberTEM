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
    | "scanSize"
    | "detectorSizeRaw"
    | "cropDetectorTo"> & {
    tileshape: string,
    scanSize: string
    detectorSizeRaw: string,
    cropDetectorTo: string,
};

type FormValues = DatasetParamsRawForForm

type MergedProps = FormikProps<FormValues> & OpenFormProps<DatasetParamsRaw> & {
    initial: DatasetParamsRaw,
};

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
                <label htmlFor="scanSize">Scan Size:</label>
                <input type="text" name="scanSize" value={values.scanSize}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="dtype">Datatype (uint16, uint32, float32, float64, ...):</label>
                <input type="text" name="dtype" value={values.dtype}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>

            <Form.Field>
                <label htmlFor="detectorSizeRaw">Detector Size (as in the file):</label>
                <input type="text" name="detectorSizeRaw" value={values.detectorSizeRaw}
                    onChange={handleChange} onBlur={handleBlur} />
            </Form.Field>
            <Form.Field>
                <label htmlFor="cropDetectorTo">Detector Size Crop:</label>
                <input type="text" name="cropDetectorTo" value={values.cropDetectorTo}
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
        detectorSizeRaw: getInitial("detectorSizeRaw", "130, 128", initial),
        cropDetectorTo: getInitial("cropDetectorTo", "128, 128", initial),
        scanSize: getInitial("scanSize", "256, 256", initial),
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
            scanSize: parseNumList(values.scanSize),
            detectorSizeRaw: parseNumList(values.detectorSizeRaw),
            cropDetectorTo: parseNumList(values.cropDetectorTo),
        });
    }
})(RawFileParamsForm);
