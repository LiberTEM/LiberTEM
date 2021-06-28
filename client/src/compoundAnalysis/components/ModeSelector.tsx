import * as React from "react";
import { Dropdown, DropdownProps } from "semantic-ui-react";


type ModeOptions = Array<{
    text: string;
    value: string;
}>;

interface ModeSelectorProps {
    modes: ModeOptions,
    currentMode: string,
    onModeChange: (mode: string) => void,
    label: string,
}

const ModeSelector: React.FunctionComponent<ModeSelectorProps> = ({
    modes, currentMode, onModeChange, label
}) => {

    const onChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        if(data.value !== undefined && typeof data.value === "string") {
            onModeChange(data.value);
        }
    }

    return (
        <>
            <div>
                {label}:{' '}
                <Dropdown
                    inline
                    options={modes}
                    value={currentMode}
                    onChange={onChange}
                />
            </div>
        </>
    )
}

export default ModeSelector;