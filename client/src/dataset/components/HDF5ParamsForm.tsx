import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Dropdown, DropdownProps, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsHDF5, DatasetTypes } from "../../messages";
import { getInitial, getInitialName, parseNumList, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

type DatasetParamsHDF5ForForm = Omit<DatasetParamsHDF5, "path" | "type" | "tileshape"> & { tileshape: string, };

type MergedProps = FormikProps<DatasetParamsHDF5ForForm> & OpenFormProps<DatasetParamsHDF5>;

const HDF5ParamsForm: React.SFC<MergedProps> = ({
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
    setFieldValue,
}) => {

    const dsPathOptions = values.dataset_paths.map(dsPath => ({ key: dsPath, text: dsPath, value: dsPath }));

    // semantic-ui requires value to be set manually on option selection
    const onDSPathChange = (e: React.SyntheticEvent, result: DropdownProps) => {
      const { value } = result;
      if (value) {
        setFieldValue("ds_path", value.toString());
      }
    };

    let dsPathInput;
    const isTimeOut = (values.dataset_paths.length === 0 ) ? true : false;
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
            <Form.Field>
                <label htmlFor="id_tileshape">Tileshape:</label>
                <ErrorMessage name="tileshape" />
                <Field name="tileshape" id="id_tileshape" />
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
            <Button onClick={onCancel} >Cancel</Button>
            <Button type="button" onClick={handleReset}>Reset</Button>
        </Form>
    )
}

export default withValidation<DatasetParamsHDF5, DatasetParamsHDF5ForForm>({
    mapPropsToValues: ({path, initial }) => ({
        name: getInitialName("name",path,initial),
        dataset_paths: getInitial("dataset_paths", [], initial),
        tileshape: getInitial("tileshape", "1, 8, 128, 128", initial).toString(),
        ds_path: getInitial("ds_path", "", initial),
    }),
    formToJson: (values, path) => {
        return {
            path,
            type: DatasetTypes.HDF5,
            name: values.name,
            ds_path: values.ds_path,
            dataset_paths: values.dataset_paths,
            tileshape: parseNumList(values.tileshape),
        };
    },
    type: DatasetTypes.HDF5,
})(HDF5ParamsForm);
