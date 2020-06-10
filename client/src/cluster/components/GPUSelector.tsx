import * as React from "react";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { ConfigState } from "../../config/reducers";

interface GPUSelectorProps {
    config: ConfigState,
    name: string,
    setFieldValue: (name: string, value: any) => void,
}


export const GPUSelector: React.FC<GPUSelectorProps> = ({ config, name, setFieldValue }) => {
    const options = config.devices.cudas.map(id => {
        return { key: id, value: id, text: `GPU ${id}` };
    });
    const myHandleChange = (e: React.ChangeEvent<any>, data: DropdownProps) => {
        setFieldValue(name, data.value);
    }
    return (
        <Dropdown onChange={myHandleChange}
           placeholder='Select CUDA devices'
           fluid={true} multiple={true} selection={true}
           options={options} />
    );
};
