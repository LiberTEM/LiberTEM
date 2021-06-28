import * as React from "react";
import { Dropdown, DropdownItemProps } from "semantic-ui-react";
import { getEnumValues } from "../../../helpers";
import { AnalysisTypes } from "../../../messages";
import { getMetadata } from "../../getMetadata";

const analysisTypeKeys = getEnumValues(AnalysisTypes);
const analysisTypeOptions = analysisTypeKeys.filter(t => getMetadata(t).component !== undefined).map(t => ({
    text: getMetadata(t).title,
    value: AnalysisTypes[t],
}));

interface AnalysisSelectProps {
    onClick: (analysisType: AnalysisTypes) => void,
    label: string,
}

const selectData = (fn: (v: AnalysisTypes) => void) => (e: React.MouseEvent, data: DropdownItemProps) => fn(data.value as AnalysisTypes)

const AnalysisSelect: React.FC<AnalysisSelectProps> = ({ onClick, label }) => (
    <Dropdown text={label} icon='add' floating labeled button className='icon'>
        <Dropdown.Menu>
            <Dropdown.Header content='implemented analyses' />
            {analysisTypeOptions.map(option => <Dropdown.Item key={option.value} onClick={selectData(onClick)} {...option} />)}
        </Dropdown.Menu>
    </Dropdown>
);

export default AnalysisSelect;