import * as React from "react";
import { Table } from "semantic-ui-react";
import { DatasetOpen, DiagElemMsg } from "../../messages";
import DatasetParams from "./DatasetParams";

interface DatasetInfoProps {
    dataset: DatasetOpen,
}

const renderValue = (elem: DiagElemMsg) => {
    if (elem.value instanceof Array) {
        return <DiagElems diagnostics={elem.value} />
    } else {
        return elem.value;
    }
}

const DiagElem: React.FC<{ elem: DiagElemMsg }> = ({ elem }) => (
    <Table.Row>
        <Table.Cell>{elem.name}</Table.Cell>
        <Table.Cell>{renderValue(elem)}</Table.Cell>
    </Table.Row>
);

const DiagElems: React.FC<{ diagnostics: DiagElemMsg[] }> = ({ diagnostics }) => {
    if (diagnostics.length === 0) {
        return null;
    }
    return (
        <Table>
            <Table.Header>
                <Table.Row>
                    <Table.HeaderCell>Name</Table.HeaderCell>
                    <Table.HeaderCell>Value</Table.HeaderCell>
                </Table.Row>
            </Table.Header>
            <Table.Body>
                {diagnostics.map((elem, idx) => (
                    <DiagElem elem={elem} key={idx} />
                ))}
            </Table.Body>
        </Table>
    );
}

const DatasetInfo: React.FC<DatasetInfoProps> = ({ dataset }) => (
    <>
        <DatasetParams dataset={dataset} />
        <DiagElems diagnostics={dataset.diagnostics} />
    </>
);

export default DatasetInfo;