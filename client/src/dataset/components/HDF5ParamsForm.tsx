import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Dropdown, DropdownProps, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetInfoHDF5, DatasetParamsHDF5, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
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

const HDF5ParamsForm: React.SFC<MergedProps> = ({
    values,
    info,
    touched,
    errors,
    dirty,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    handleReset,
    onCancel,
    setFieldValue,
}) => {

    const dsPathOptions = info?.dataset_paths.map(dsPath => ({ key: dsPath, text: dsPath, value: dsPath }));

    // semantic-ui requires value to be set manually on option selection
    const onDSPathChange = (e: React.SyntheticEvent, result: DropdownProps) => {
      const { value } = result;
      if (value) {
        setFieldValue("ds_path", value.toString());
      }
    };

    let dsPathInput;
    const isTimeOut = (info?.dataset_paths.length === 0 ) ? true : false;

    if (isTimeOut) {
      dsPathInput = <Field name="ds_path" id="id_ds_path" />;
    } else {
      dsPathInput = <Dropdown name="ds_path" id="id_ds_path" placeholder="Select dataset" fluid={true} search={true} selection={true} defaultValue={values.ds_path} onChange={onDSPathChange} options={dsPathOptions} />;
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
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button onClick={onCancel} >Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsHDF5, DatasetParamsHDF5ForForm, DatasetInfoHDF5>({
    mapPropsToValues: ({ path, initial }) => ({
        name: getInitialName("name", path, initial),
        ds_path: getInitial("ds_path", "", initial),
        nav_shape: getInitial("nav_shape", "", initial).toString(),
        sig_shape: getInitial("sig_shape", "", initial).toString(),
        sync_offset: getInitial("sync_offset", 0, initial),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.HDF5,
            name: values.name,
            ds_path: values.ds_path,
            nav_shape: parseNumList(values.nav_shape),
            sig_shape: parseNumList(values.sig_shape),
            sync_offset: values.sync_offset,
        };
    },
    type: DatasetTypes.HDF5,
})(HDF5ParamsForm);
