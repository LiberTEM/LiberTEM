import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Dropdown, DropdownProps, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoHDF5, DatasetInfoHDF5Item, DatasetParamsHDF5, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, adjustShapeWithBounds, parseShapeInCommaSeparatedString, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

type DatasetParamsHDF5ForForm = Omit<DatasetParamsHDF5,
    "type"
    | "path"
    | "nav_shape"
    | "sig_shape"> & {
        nav_shape: string,
        sig_shape: string,
};

type MergedProps = FormikProps<DatasetParamsHDF5ForForm> & OpenFormProps<DatasetParamsHDF5, DatasetInfoHDF5>;

const HDF5ParamsForm: React.FC<MergedProps> = ({
    values,
    info,
    isSubmitting,
    handleSubmit,
    handleReset,
    onCancel,
    setFieldValue,
}) => {
    const dsItemsByPath: {
        [k: string]: DatasetInfoHDF5Item
    } = {};
    info?.datasets?.forEach(dsItem => dsItemsByPath[dsItem.path] = dsItem);

    const dsPathOptions = info?.datasets?.map(dsItem => {
        const shape = dsItem.shape.join(",")
        const opts: string[] = [];

        if(dsItem.chunks !== null) {
            opts.push('chunked');
        }

        if(dsItem.compression !== null) {
            opts.push(`compression: ${dsItem.compression}`);
        }

        const text = `${dsItem.path} (shape: (${shape}), ${opts.join(", ")})`;
        return {
            text,
            key: dsItem.path,
            value: dsItem.path,
        };
    });

    // semantic-ui requires value to be set manually on option selection
    const onDSPathChange = (e: React.SyntheticEvent, result: DropdownProps) => {
      const { value } = result;
      if (value) {
        setFieldValue("ds_path", value.toString());
      }
    };

    let dsPathInput;
    const pathsLength = info?.datasets?.length
    const isTimeOut = pathsLength === 0 || pathsLength === undefined;

    if (isTimeOut) {
      dsPathInput = <Field name="ds_path" id="id_ds_path" />;
    } else {
        dsPathInput = <Dropdown name="ds_path" id="id_ds_path" placeholder="Select dataset" fluid search selection defaultValue={values.ds_path} onChange={onDSPathChange} options={dsPathOptions} />;
    }

    let warning = null;
    const selectedItem = dsItemsByPath[values.ds_path];
    if (selectedItem && selectedItem.compression) {
        warning = (
            <p><strong style={{ color: "red" }}>Loading compressed HDF5, performance can be worse than with other formats</strong></p>
        );
    }

    return (
        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="id_name">Name:</label>
                <ErrorMessage name="name" />
                <Field name="name" id="id_name" />
            </Form.Field>
            <Form.Field>
                <label htmlFor="id_ds_path">HDF5 Dataset Path:</label>
                <ErrorMessage name="ds_path" />
                {dsPathInput}
            </Form.Field>
            {warning}
            <Button primary type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button onClick={onCancel} >Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsHDF5, DatasetParamsHDF5ForForm, DatasetInfoHDF5>({
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        ds_path: getInitial("ds_path", "", initial),
        nav_shape: adjustShapeWithBounds(getInitial("nav_shape", "", initial).toString(), "nav"),
        sig_shape: adjustShapeWithBounds(getInitial("sig_shape", "", initial).toString(), "sig"),
        sync_offset: getInitial("sync_offset", 0, initial),
    }),
    formToJson: (values, path) => ({
        path,
        type: DatasetTypes.HDF5,
        name: values.name,
        ds_path: values.ds_path,
        nav_shape: parseShapeInCommaSeparatedString(values.nav_shape),
        sig_shape: parseShapeInCommaSeparatedString(values.sig_shape),
        sync_offset: values.sync_offset,
    }),
    type: DatasetTypes.HDF5,
})(HDF5ParamsForm);
