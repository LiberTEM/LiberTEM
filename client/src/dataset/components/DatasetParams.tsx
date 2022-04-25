import * as React from "react";
import { Table } from "semantic-ui-react";
import { DatasetFormParams, DatasetState } from "../../messages";

interface DatasetProps {
    dataset: DatasetState
}

const renderParamValue = (value: any[] | string) => {
    if (value instanceof Array) {
        return `(${value.join(",")})`;
    } else {
        return value.toString();
    }
}

const renderRow = (param: any[] | string, key: string, idx: number) => (
    <Table.Row key={idx}>
        <Table.Cell>{key}</Table.Cell>
        <Table.Cell>{renderParamValue(param)}</Table.Cell>
    </Table.Row>
);

const renderParams = (params: DatasetFormParams) => 
    Object.entries(params).map(([key, param], idx) => {
        if (param && (typeof param === "string" || param instanceof Array)) {
            return renderRow(param, key, idx);
        }
    });

const DatasetParams: React.FC<DatasetProps> = ({ dataset }) => (
    <Table>
        <Table.Header>
            <Table.Row>
                <Table.HeaderCell>Parameter</Table.HeaderCell>
                <Table.HeaderCell>Value</Table.HeaderCell>
            </Table.Row>
        </Table.Header>
        <Table.Body>
            {renderParams(dataset.params)}
        </Table.Body>
    </Table>
);


export default DatasetParams;
