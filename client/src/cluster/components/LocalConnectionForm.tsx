import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { ClusterTypes, ConnectRequestLocalCluster } from "../../messages";

type FormValues = Omit<ConnectRequestLocalCluster, "type">;

interface FormProps {
    onSubmit: (params: ConnectRequestLocalCluster) => void,
}

type MergedProps = FormikProps<FormValues> & FormProps;

const LocalConnectionForm: React.SFC<MergedProps> = ({
    values,
    touched,
    errors,
    dirty,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    handleReset,
}) => {
    return (
        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="numWorkers">Number of Workers:</label>
                <input type="number" name="numWorkers" value={values.numWorkers}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.numWorkers && touched.numWorkers && errors.numWorkers}
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Connect</Button>
        </Form>
    )
}

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: () => ({
        numWorkers: 4,  // TODO: sensible default based on number of cores and hyperthreading
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: ClusterTypes.LOCAL,
            ...values,
        });
    }
})(LocalConnectionForm);
