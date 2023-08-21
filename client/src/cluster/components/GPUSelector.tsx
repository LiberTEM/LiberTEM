import * as React from "react";
import { Form, Message } from "semantic-ui-react";
import { ConfigState } from "../../config/reducers";

interface GPUSelectorProps {
    config: ConfigState,
    name: string,
    value: Record<number, number | string>
    setFieldValue: (name: string, value: any) => void,
}


export const GPUSelector: React.FC<GPUSelectorProps> = ({
    config, name, value, setFieldValue,
}) => {
    const hasCupy = config.devices.has_cupy;
    const cudaIds = config.devices.cudas;
    const haveCudaDevices = cudaIds.length > 0;
    const disabled = !hasCupy || !haveCudaDevices;
    const showWarning = !hasCupy && haveCudaDevices;

    const onNumChange = (id: number, event: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = {...value};
        const parsed = parseInt(event.target.value, 10);
        if (!isNaN(parsed)) {
            newValue[id] = parsed;
        } else {
            newValue[id] = event.target.value.toString();
        }

        setFieldValue(name, newValue);
    }

    const onBlur = (id: number, event: React.FocusEvent<HTMLInputElement>) => {
        if(isNaN(parseInt(event.target.value, 10))) {
            const newValue = {...value};
            newValue[id] = 0;
            setFieldValue(name, newValue);
        }
    }

    return (
        <>
            {showWarning ?
                <Message warning visible>
                    <Message.Header>No cupy installation found</Message.Header>
                    <p>
                        To make use of the built-in GPU support, make
                        sure to install <a href="https://cupy.dev/" rel="noreferrer noopener" target="_blank">cupy</a>
                    </p>
                </Message> : ''
            }
            {!haveCudaDevices ? 
            <Message info visible>
                <Message.Header>No CUDA devices found</Message.Header>
                <p>
                    GPU support requires one or more CUDA compatible devices, and working
                    CUDA and cupy installations. 
                </p>
            </Message>
            : ''}
            <ul style={{paddingLeft: 0}}>
                {cudaIds.map(id => (
                    value[id] !== undefined ?
                    <li key={id} style={{listStyleType: 'none'}}>
                        <Form.Field>
                            <label>
                                Number of workers for CUDA device {id}:
                                <input
                                    disabled={disabled} type="number" min={0} value={value[id]}
                                    onChange={e => onNumChange(id, e)}
                                    onBlur={e => onBlur(id, e)} />
                            </label>
                        </Form.Field>
                    </li> : ''
                ))}
            </ul>
        </>
    );
};
