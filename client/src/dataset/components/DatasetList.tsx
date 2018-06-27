import * as React from "react";
import { connect } from "react-redux";
import { Button, Icon, Label } from "semantic-ui-react";
import * as uuid from "uuid/v4";
import { RootReducer } from "../../store";
import * as datasetActions from "../actions";
import { DatasetsState } from "../types";
import Dataset from "./Dataset";

interface DatasetListProps {
    datasets: DatasetsState
}

const mapStateToProps = (state: RootReducer) => {
    return {
        datasets: state.dataset,
    };
}

const mapDispatchToProps = {
    createDataset: datasetActions.Actions.create,
}

type MergedProps = DatasetListProps & DispatchProps<typeof mapDispatchToProps>

const DatasetList: React.SFC<MergedProps> = ({ createDataset, datasets }) => {
    const loadDatasets = () => {
        createDataset({
            id: uuid(),
            name: "test dataset",
            path: "/test/index.json",
            tileshape: [1, 8, 128, 128],
            type: "HDFS",
        })

        createDataset({
            id: uuid(),
            name: "e field mapping acquisition 8",
            path: "/e-field-acquisition_8_0tilt_0V/index.json",
            tileshape: [1, 8, 128, 128],
            type: "HDFS",
        })

        createDataset({
            id: uuid(),
            name: "e field mapping acquisition 10",
            path: "/e-field-acquisition_10_0tilt_40V/index.json",
            tileshape: [1, 8, 128, 128],
            type: "HDFS",
        })
    }
    return (
        <>
            <Button as='div' labelPosition='right' onClick={loadDatasets}>
                <Button icon={true}>
                    <Icon name='folder open outline' />
                </Button>
                <Label as='a' basic={true}>
                    Load Dataset
                </Label>
            </Button>

            {datasets.ids.map(dsId => <Dataset dataset={datasets.byId[dsId]} key={dsId} />)}
        </>
    );
}

export default connect(mapStateToProps, mapDispatchToProps)(DatasetList);