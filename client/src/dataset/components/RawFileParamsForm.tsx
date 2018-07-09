import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsRaw, DatasetTypes } from "../../messages";

// some fields have different types in the form vs. in messages
type DatasetParamsRawForForm = Omit<DatasetParamsRaw,
    "type"
    | "tileshape"
    | "scanSize"
    | "detectorSizeRaw"
    | "cropDetectorTo"> & {
    tileshape: string,
    scanSize: string
    detectorSizeRaw: string,
    cropDetectorTo: string,
};

type FormValues = DatasetParamsRawForForm

interface FormProps {
    onSubmit: (params: DatasetParamsRaw) => void
    onCancel: () => void,
}

type MergedProps = FormikProps<FormValues> & FormProps;

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
                <label htmlFor="path">Path:</label>
                <input type="text" name="path" value={values.path}
                    onChange={handleChange} onBlur={handleBlur} />
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

function parseNumList(nums: string) {
    return nums.split(",").map(part => +part);
}

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: () => ({
        name: "",
        tileshape: "1, 8, 128, 128",
        detectorSizeRaw: "130, 128",
        cropDetectorTo: "128, 128",
        scanSize: "256, 256",
        dtype: "float32",
        path: "",
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: DatasetTypes.RAW,
            name: values.name,
            path: values.path,
            dtype: values.dtype,
            tileshape: parseNumList(values.tileshape),
            scanSize: parseNumList(values.scanSize),
            detectorSizeRaw: parseNumList(values.detectorSizeRaw),
            cropDetectorTo: parseNumList(values.cropDetectorTo),
        });
    }
})(RawFileParamsForm);
