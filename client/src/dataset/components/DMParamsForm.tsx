import { ErrorMessage, Field, FormikProps } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { DatasetParamsDM, DatasetTypes } from "../../messages";
import { getInitialName, withValidation } from "../helpers";
import { OpenFormProps } from "../types";

// some fields have different types in the form vs. in messages
type DatasetParamsDMForForm = Omit<DatasetParamsDM,
  "path"
  | "type"
>;

type MergedProps = FormikProps<DatasetParamsDMForForm> & OpenFormProps<DatasetParamsDM>;

const DMParamsForm: React.SFC<MergedProps> = ({
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
      <Button primary={true} type="submit" disabled={isSubmitting}>Load Dataset</Button>
      <Button type="button" onClick={onCancel}>Cancel</Button>
      <Button type="button" onClick={handleReset}>Reset</Button>
    </Form>
  )
}

export default withValidation<DatasetParamsDM, DatasetParamsDMForForm>({
  mapPropsToValues: ({ path, initial }) => ({
    name: getInitialName("name", path, initial),
  }),
  formToJson: (values, path) => {
    return {
      path,
      type: DatasetTypes.DM,
      name: values.name,
    };
  },
  type: DatasetTypes.DM,
})(DMParamsForm);
