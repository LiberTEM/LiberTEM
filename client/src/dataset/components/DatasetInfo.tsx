import * as React from "react";
import { DatasetState } from "../../messages";
import DatasetParams from "./DatasetParams";

interface DatasetInfoProps {
    dataset: DatasetState,
}

const DatasetInfo: React.SFC<DatasetInfoProps> = ({ dataset }) => {
    return (
        <>
            <DatasetParams dataset={dataset} />
        </>
    );
}

export default DatasetInfo;