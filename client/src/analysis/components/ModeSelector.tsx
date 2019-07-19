import * as React from "react";
import { Dropdown, DropdownProps } from "semantic-ui-react";


type ModeOptions = Array<{
    text: string;
    value: any;
}>;

interface ModeSelectorProps {
    modes: ModeOptions,
    currentMode: any,
    onModeChange: (mode: any) => void,
}

const ModeSelector: React.FunctionComponent<ModeSelectorProps> = ({
    modes, currentMode, onModeChange,
}) => {

    const onChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        onModeChange(data.value)
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