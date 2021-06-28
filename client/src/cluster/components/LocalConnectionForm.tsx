import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Button, Form, Icon, Modal } from "semantic-ui-react";
import { ConfigState } from "../../config/reducers";
import { Omit } from "../../helpers/types";
import { ClusterTypes, ConnectRequestLocalCluster } from "../../messages";
import { GPUSelector } from "./GPUSelector";

type FormValues = Omit<ConnectRequestLocalCluster, "type"> & {
    cudas: number[],
};

interface FormProps {
    onSubmit: (params: ConnectRequestLocalCluster) => void,
    config: ConfigState,
}

type MergedProps = FormikProps<FormValues> & FormProps;

const LocalConnectionForm: React.FC<MergedProps> = ({
    config,
    values,
    touched,
    errors,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    setFieldValue,
}) => {
    // default value:
    React.useEffect(() => {
        setFieldValue("cudas", config.devices.cudas);
    }, [config.devices.cudas, setFieldValue]);

    return (
        <Form onSubmit={handleSubmit}>
            <Form.Field>
                <label htmlFor="numWorkers">Number of Workers:</label>
                <input type="number" name="numWorkers" value={values.numWorkers}
                    onChange={handleChange}
                    onBlur={handleBlur} />
                {errors.numWorkers && touched.numWorkers && errors.numWorkers}
            </Form.Field>
            <Form.Field>
                <label htmlFor="cudas">
                    CUDA devices to use{' '}
                    <Modal trigger={<Icon name="info circle" link/>}>
                        <Modal.Header>
                            CUDA information
                        </Modal.Header>
                        <Modal.Content>
                            <p>
                                For some operations, LiberTEM can automatically make use of your graphics card,
                                if it supports CUDA.
                            </p>
                            <ul>
                                <li>Number of CUDA devices found: {config.devices.cudas.length}</li>
                                <li>cupy installation found:{' '}{config.devices.has_cupy ? 'Yes' : 'No'}</li>
                            </ul>
                            <p>cupy needs to be installed to make use of any CUDA devices on your system. Also,
                                the matching cuda libraries and graphics drivers need to be installed. Please
                                refer to{' '}
                                <a href="https://docs.cupy.dev/en/stable/install.html" target="_blank" rel="noopener noreferrer">
                                    the cupy documentation
                                </a>{' '}
                                for more information.
                            </p>
                        </Modal.Content>
                    </Modal>
                </label>
                <GPUSelector name="cudas" value={values.cudas} config={config} setFieldValue={setFieldValue} />
            </Form.Field>
            <Button primary type="submit" disabled={isSubmitting}>Connect</Button>
        </Form>
    )
}

export default withFormik<FormProps, FormValues>({
    mapPropsToValues: (ownProps: FormProps) => ({
        numWorkers: ownProps.config.localCores,
        cudas: [],
    }),
    handleSubmit: (values, formikBag) => {
        const { onSubmit } = formikBag.props;
        onSubmit({
            type: ClusterTypes.LOCAL,
            ...values,
        });
    }
})(LocalConnectionForm);
