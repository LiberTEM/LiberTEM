import * as React from "react";
import { Dropdown, DropdownItemProps } from "semantic-ui-react";
import { getEnumValues } from "../../helpers";
import { DatasetTypes } from "../../messages";


const datasetTypeKeys = getEnumValues(DatasetTypes);
const datasetTypeOptions = datasetTypeKeys.map(t => ({
    // text: DatasetTypeMetadata[DatasetTypes[t as any]].short,
    text: DatasetTypes[t as any],
    value: DatasetTypes[t as any],
}));

interface DatasetTypeSelectProps {
    onClick: (datasetType: DatasetTypes) => void,
    label: string,
}

const selectData = (fn: (v: DatasetTypes) => void) => (e: React.MouseEvent, data: DropdownItemProps) => {
    return fn(data.value as DatasetTypes);
}

const DatasetTypeSelect: React.SFC<DatasetTypeSelectProps> = ({ onClick, label }) => {
    return (
        <Dropdown text={label} icon='add' floating={true} labeled={true} button={true} className='icon'>
            <Dropdown.Menu>
                <Dropdown.Header content='dataset types' />
                {datasetTypeOptions.map(option => <Dropdown.Item key={option.value} onClick={selectData(onClick)} {...option} />)}
            </Dropdown.Menu>
        </Dropdown>
    );
}

export default DatasetTypeSelect;