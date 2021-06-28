import { FormikProps, withFormik } from "formik";
import * as React from "react";
import { Form, Input } from "semantic-ui-react";

interface FormValues {
    path: string,
}

interface PathInputProps {
    onChange: (path: string) => void,
    initialPath: string,
}

type MergedProps = FormikProps<FormValues> & PathInputProps;

const PathInput: React.FC<MergedProps> = ({
    values,
    handleChange,
    handleBlur,
    handleSubmit,
}) => (
    <Form onSubmit={handleSubmit} style={{ flexGrow: 1 }}>
        <Form.Field>
            <Input autoComplete="off" onBlur={handleBlur} onChange={handleChange} value={values.path} name="path" />
        </Form.Field>
    </Form>
);

export default withFormik<PathInputProps, FormValues>({
    mapPropsToValues: ({ initialPath }) => ({
        path: initialPath,
    }),
    handleSubmit: (values, formikBag) => {
        const { onChange } = formikBag.props;
        onChange(values.path);
    },
    enableReinitialize: true,
})(PathInput);