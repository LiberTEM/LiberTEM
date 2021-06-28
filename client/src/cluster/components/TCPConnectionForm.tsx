import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { ConfigState } from "../../config/reducers";
import { Omit } from "../../helpers/types";
import { ClusterTypes, ConnectRequestTCP } from "../../messages";

type FormValues = Omit<ConnectRequestTCP, "type">;

interface FormProps {
    onSubmit: (params: ConnectRequestTCP) => void,
    config: ConfigState,
}

type MergedProps = FormikProps<FormValues> & FormProps;

const TCPConnectionForm: React.FC<MergedProps> = ({
    values,
    touched,
    errors,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
}) => (
    <Form onSubmit={handleSubmit}>
        <Form.Field>
            <label htmlFor="address">Scheduler URI</label>
            <input type="text" name="address" value={values.address}
                onChange={handleChange}
                onBlur={handleBlur} />
            {errors.address && touched.address && errors.address}
        </Form.Field>
        <Button primarytype="submit" disabled={isSubmitting}>Connect</Button>
    </Form>
)

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: (ownProps: FormProps) => ({
        address: ownProps.config.lastConnection.address,
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: ClusterTypes.TCP,
            ...values
        });
    }
})(TCPConnectionForm);
