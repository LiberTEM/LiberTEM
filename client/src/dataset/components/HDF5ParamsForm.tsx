import { ErrorMessage, Field, FormikProps, FormikErrors, FormikValues } from "formik";
import * as React from "react";
import { Button, Dropdown, DropdownProps, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoHDF5, DatasetInfoHDF5Item, DatasetParamsHDF5, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, adjustShapeWithBounds, parseShapeInCommaSeparatedString, withValidation, validateSyncOffsetAndSigShape } from "../helpers";
import { OpenFormProps } from "../types";
import Reshape from "./Reshape";

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
    setFieldTouched,
}) => {
    const dsItemsByPath: {
        [k: string]: DatasetInfoHDF5Item
    } = {};
    info?.datasets?.forEach(dsItem => dsItemsByPath[dsItem.path] = dsItem);

    const dsPathOptions = info?.datasets?.map(dsItem => {
        const rawNavShape = dsItem.raw_nav_shape.join(", ")
        const sigShape = dsItem.sig_shape.join(", ")
        const opts: string[] = [];

        if(dsItem.chunks !== null) {
            opts.push('chunked');
        }

        if(dsItem.compression !== null) {
            opts.push(`compression: ${dsItem.compression}`);
        }

        const text = `${dsItem.path} (nav_shape: (${rawNavShape}), sig_shape: (${sigShape}) ${opts.join(", ")})`;
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
          const dsPath = value.toString()
          setFieldValue("ds_path", dsPath);
          const dsInfo = dsItemsByPath[dsPath]
          if (dsInfo === undefined) {
              setFieldValue("nav_shape", "1,1")
              setFieldValue("sig_shape", "1,1")
          } else {
              setFieldValue("nav_shape", dsInfo.nav_shape.toString())
              setFieldValue("sig_shape", dsInfo.sig_shape.toString())
          }
          setFieldTouched('nav_shape', false)
          setFieldTouched('sig_shape', false)
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
            <Reshape navShape={values.nav_shape} sigShape={values.sig_shape} syncOffset={values.sync_offset} hideInfo setFieldValue={setFieldValue} setFieldTouched={setFieldTouched} />
            <Button primary type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button onClick={onCancel} >Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

const getInfoItemForDSPath = (dsPath: string, info?: DatasetInfoHDF5): DatasetInfoHDF5Item => {
    const dsItemsByPath: {
        [k: string]: DatasetInfoHDF5Item
    } = {};
    info?.datasets?.forEach(dsItem => dsItemsByPath[dsItem.path] = dsItem);
    // I think this will fail if datasets is undefined or empty
    // yet the only possible value for ds_path comes from the dropdown
    // menu which is itself defined by info.datasets so by definition
    // it is valid!
    return dsItemsByPath[dsPath]
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
    customValidation: (values, { info }) => {
        const dsInfo = getInfoItemForDSPath(values.ds_path, info)
        if (dsInfo === undefined) {
            const unknownErrors: FormikErrors<FormikValues> = {};
            return unknownErrors
        }
        return validateSyncOffsetAndSigShape(
            dsInfo.sig_shape,
            values.sig_shape,
            values.sync_offset,
            dsInfo.image_count,
            true,
        )
    },
    type: DatasetTypes.HDF5,
})(HDF5ParamsForm);
