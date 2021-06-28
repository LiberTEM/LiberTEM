import * as React from "react";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { getEnumValues } from "../../helpers";
import { DatasetTypes } from "../../messages";


const datasetTypeKeys = getEnumValues(DatasetTypes);
const datasetTypeOptions = datasetTypeKeys.map(t => ({
    // text: DatasetTypeMetadata[DatasetTypes[t as any]].short,
    text: DatasetTypes[t],
    value: DatasetTypes[t],
}));

interface DatasetTypeSelectProps {
    onClick: (e: React.SyntheticEvent, data: DropdownProps) => void,
    currentType: DatasetTypes,
}

const DatasetTypeSelect: React.FC<DatasetTypeSelectProps> = ({ currentType, onClick }) => (
    <>
        <Dropdown
            inline
            options={datasetTypeOptions}
            value={currentType}
            onChange={onClick}
        />
    </>
);

export default DatasetTypeSelect;