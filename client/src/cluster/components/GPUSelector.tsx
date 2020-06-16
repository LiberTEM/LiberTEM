import * as React from "react";
import { Dropdown, DropdownProps, Message } from "semantic-ui-react";
import { ConfigState } from "../../config/reducers";

interface GPUSelectorProps {
    config: ConfigState,
    name: string,
    setFieldValue: (name: string, value: any) => void,
}


export const GPUSelector: React.FC<GPUSelectorProps> = ({
    config, name, setFieldValue,
}) => {
    const options = config.devices.cudas.map(id => {
        return { key: id, value: id, text: `GPU ${id}` };
    });
    const defaultValue = config.devices.cudas;
    const myHandleChange = (e: React.ChangeEvent<any>, data: DropdownProps) => {
        setFieldValue(name, data.value);
    }
    const disabled = !config.devices.has_cupy;
    return (
        <>
            {disabled ?
                <Message warning={true} visible={true}>
                    <Message.Header>No cupy installation found</Message.Header>
                    <p>
                        To make use of the built-in GPU support, make
                        sure to install <a href="https://cupy.chainer.org/" rel="noreferrer noopener" target="_blank">cupy</a>
                    </p>
                </Message> : ''
            }
            <Dropdown onChange={myHandleChange}
                disabled={disabled}
                placeholder='Select CUDA devices'
                fluid={true} multiple={true} selection={true}
                defaultValue={defaultValue}
                options={options} />
        </>
    );
};
