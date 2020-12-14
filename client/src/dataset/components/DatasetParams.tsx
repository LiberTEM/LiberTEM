import * as React from "react";
import { Table } from "semantic-ui-react";
import { DatasetState } from "../../messages";

interface DatasetProps {
    dataset: DatasetState
}

const renderParamValue = (value: any) => {
    if (value instanceof Array) {
        return `(${value.join(",")})`;
    } else {
        return value;
    }
}

const renderRow = (params: any, key:string, idx: number) => {
    return (
        <Table.Row key={idx}>
            <Table.Cell>{key}</Table.Cell>
            <Table.Cell>{renderParamValue(params[key])}</Table.Cell>
        </Table.Row>
    );
}

const renderParams = (params: any) => {
    return Object.keys(params).map((key: string, idx: number) => {
        if(typeof params[key] === 'object' && !(params[key] instanceof Array)) {
            return Object.keys(params[key]).map((objKey: string, keyIdx: number) => {
                return renderRow(params[key], objKey, keyIdx);
            });
        } else {
            return renderRow(params, key, idx);
        }
    })
}


const DatasetParams: React.SFC<DatasetProps> = ({ dataset }) => {
    return (
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
}


export default DatasetParams;
