import React from "react"
import { Dropdown, DropdownProps } from "semantic-ui-react";
import { DatasetTypeInfo, IOBackendId, IOBackendMetadata } from "../../messages";

export interface BackendSelectionDropdownProps {
    value?: string,
    datasetTypeInfo: DatasetTypeInfo,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
}

const BackendSelectionDropdown: React.FC<BackendSelectionDropdownProps> = ({
    value,
    setFieldValue,
    datasetTypeInfo,
}) => {
    const handleChange = (e: React.SyntheticEvent, data: DropdownProps) => {
        const backendId = data.value as IOBackendId;
        setFieldValue("io_backend", backendId);
    };

    const backendOptions = datasetTypeInfo.supported_io_backends.map(backendId => ({
        text: IOBackendMetadata[backendId].label,
        value: backendId,
    }));

    const defaultBackend = datasetTypeInfo.default_io_backend
    const defaultValue = defaultBackend ? defaultBackend : "";

    return (
        <>
            <Dropdown
            selection
            options={backendOptions}
            value={value ? value : defaultValue}
            onChange={handleChange} />
        </>
    );
}

export default BackendSelectionDropdown;
