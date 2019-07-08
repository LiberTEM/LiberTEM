import * as React from "react";
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { AnalysisTypes } from "../../messages";

type ModeOptions = Array<{
    text: string;
    value: AnalysisTypes;
}>;

interface ModeSelectorProps {
    modes: ModeOptions,
    currentMode: AnalysisTypes,
    onModeChange: (mode: AnalysisTypes) => void,
}

const ModeSelector: React.FunctionComponent<ModeSelectorProps> = ({
    modes, currentMode, onModeChange,
}) => {

    const onChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        onModeChange(data.value as AnalysisTypes)
    }

    return (
        <>
            <div>
                Mode:{' '}
                <Dropdown
                    inline={true}
                    options={modes}
                    value={currentMode}
                    onChange={onChange}
                />
            </div>
        </>
    )
}

export default ModeSelector;