
import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { Omit } from "../../helpers/types";
import { ClusterTypes, ConnectRequestTCP } from "../../messages";

type FormValues = Omit<ConnectRequestTCP, "type">;

interface FormProps {
    onSubmit: (params: ConnectRequestTCP) => void,
}

type MergedProps = FormikProps<FormValues> & FormProps;

const TCPConnectionForm: React.SFC<MergedProps> = ({
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
                <label htmlFor="address">Scheduler URI</label>
                <input type="text" name="address" value={values.address}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.address && touched.address && errors.address}
            </Form.Field>
            <Button primary={true} type="submit" disabled={isSubmitting}>Connect</Button>
        </Form>
    )
}

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: () => ({
        address: "tcp://localhost:8786",
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: ClusterTypes.TCP,
            ...values
        });
    }
})(TCPConnectionForm);
