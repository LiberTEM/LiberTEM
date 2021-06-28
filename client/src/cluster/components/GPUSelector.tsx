import * as React from "react";
import { Dropdown, DropdownProps, Message } from "semantic-ui-react";
import { ConfigState } from "../../config/reducers";

interface GPUSelectorProps {
    config: ConfigState,
    name: string,
    value: number[],
    setFieldValue: (name: string, value: any) => void,
}


export const GPUSelector: React.FC<GPUSelectorProps> = ({
    config, name, value, setFieldValue,
}) => {
    const options = config.devices.cudas.map(id => ({ key: id, value: id, text: `GPU ${id}` }));
    const myHandleChange = (e: React.ChangeEvent<any>, data: DropdownProps) => {
        setFieldValue(name, data.value);
    }
    const hasCupy = config.devices.has_cupy;
    const haveCudaDevices = config.devices.cudas.length > 0;
    const disabled = !hasCupy || !haveCudaDevices;
    const showWarning = !hasCupy && haveCudaDevices;
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
            <Dropdown onChange={myHandleChange}
                disabled={disabled}
                placeholder='Select CUDA devices'
                fluid multiple selection
                value={value}
                options={options} />
        </>
    );
};
