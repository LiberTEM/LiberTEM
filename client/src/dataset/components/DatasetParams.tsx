import * as React from "react";
import { Table } from "semantic-ui-react";
import { DatasetState } from "../../messages";
import { isAdditionalInfo } from "../helpers";

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

const renderParams = (params: any) => {
    return Object.keys(params).map((key: string, idx: number) => {
        // Only show parameters, not additional info
        if(!isAdditionalInfo(key)) {
            return (
                <Table.Row key={idx}>
                    <Table.Cell>{key}</Table.Cell>
                    <Table.Cell>{renderParamValue(params[key])}</Table.Cell>
                </Table.Row>
            );
        }
        else {
          return null;
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
