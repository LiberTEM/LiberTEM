import * as React from "react";
import { connect } from "react-redux";
import { DatasetTypes } from "../../messages";
import { RootReducer } from "../../store";
import { DatasetsState } from "../types";
import Dataset from "./Dataset";
import DatasetOpen from "./DatasetOpen";
import DatasetTypeSelect from "./DatasetTypeSelect";

interface DatasetListProps {
    datasets: DatasetsState
}

const mapStateToProps = (state: RootReducer) => {
    return {
        datasets: state.dataset,
    };
}

type MergedProps = DatasetListProps;

interface DatasetListState {
    datasetFormOpen: boolean,
    datasetType?: DatasetTypes
}

class DatasetList extends React.Component<MergedProps, DatasetListState> {
    public state = {
        datasetFormOpen: false,
        datasetType: undefined,
    }

    public openDatasetForm = (type: DatasetTypes) => {
        this.setState({
            datasetFormOpen: true,
            datasetType: type,
        })
    };

    public closeForm = () => {
        this.setState({
            datasetFormOpen: false,
        })
    }

    public render() {
        const { datasets } = this.props;
        const { datasetFormOpen, datasetType } = this.state;

        return (
            <>
                <DatasetTypeSelect onClick={this.openDatasetForm} label='Load Dataset' />
                {datasetFormOpen && datasetType !== undefined && <DatasetOpen type={datasetType} onCancel={this.closeForm} onSubmit={this.closeForm} />}
                {datasets.ids.map(dsId => <Dataset dataset={datasets.byId[dsId]} key={dsId} />)}
            </>
        );
    }
}

/*const loadDatasets = () => {
    createDataset({
        id: uuid(),
        params: {
            name: "test dataset",
            type: DatasetTypes.HDFS,
            path: "/test/index.json",
            tileshape: [1, 8, 128, 128],
        }
    })

    createDataset({
        id: uuid(),
        params: {
            name: "e field mapping acquisition 8",
            type: DatasetTypes.HDFS,
            path: "/e-field-acquisition_8_0tilt_0V/index.json",
            tileshape: [1, 8, 128, 128],
        },
    })

    createDataset({
        id: uuid(),
        params: {
            name: "e field mapping acquisition 10",
            type: DatasetTypes.HDFS,
            path: "/e-field-acquisition_10_0tilt_40V/index.json",
            tileshape: [1, 8, 128, 128],
        }
    })

    createDataset({
        id: uuid(),
        params: {
            name: "local HDF5 test dataset",
            type: DatasetTypes.HDF5,
            path: "/home/clausen/Data/EMPAD/scan_11_x256_y256.emd",
            dsPath: "experimental/science_data/data",
            tileshape: [1, 8, 128, 128],
        }
    })
}*/


export default connect(mapStateToProps)(DatasetList);