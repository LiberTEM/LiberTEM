import * as React from "react";
import { connect, Dispatch } from 'react-redux';
import { DropdownProps, Header, Segment } from "semantic-ui-react";
import * as uuid from "uuid/v4";
import { DatasetFormParams, DatasetTypes } from '../../messages';
import { RootReducer } from "../../store";
import * as datasetActions from "../actions";
import BLOParamsForm from "./BLOParamsForm";
import DatasetTypeSelect from "./DatasetTypeSelect";
import HDF5ParamsForm from "./HDF5ParamsForm";
import HDFSParamsForm from './HDFSParamsForm';
import MIBParamsForm from "./MIBParamsForm";
import RawFileParamsForm from "./RawFileParamsForm";

const mapDispatchToProps = (dispatch: Dispatch) => {
    return {
        createDataset: (params: DatasetFormParams) => {
            dispatch(datasetActions.Actions.create({
                id: uuid(),
                params,
            }));
        },
        onCancel: () => dispatch(datasetActions.Actions.cancelOpen()),
    };
}

const mapStateToProps = (state: RootReducer) => {
    return {
        formVisible: state.openDataset.formVisible,
        formPath: state.openDataset.formPath,
    };
}


type MergedProps = ReturnType<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

interface DatasetOpenState {
    datasetType: DatasetTypes
}


class DatasetOpen extends React.Component<MergedProps, DatasetOpenState> {
    public state = {
        datasetType: DatasetTypes.RAW,
    }

    public setDatasetType = (e: React.SyntheticEvent, data: DropdownProps) => {
        const type = data.value as DatasetTypes;
        this.setState({
            datasetType: type,
        })
    }

    public render() {
        const { formPath, createDataset, onCancel } = this.props;
        const { datasetType } = this.state;

        const renderForm = (form: React.ReactNode) => {
            return (
                <Segment>
                    Type: <DatasetTypeSelect onClick={this.setDatasetType} currentType={datasetType} />
                    <Header as="h2">Open: {formPath}</Header>
                    {form}
                </Segment>
            );
        }
        if (formPath === undefined) {
            // tslint:disable-next-line:no-console
            console.error("formPath is undefined");
            return null;
        }
        switch (datasetType) {
            case DatasetTypes.HDF5: {
                return renderForm(<HDF5ParamsForm path={formPath} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.HDFS: {
                return renderForm(<HDFSParamsForm path={formPath} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.RAW: {
                return renderForm(<RawFileParamsForm path={formPath} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.MIB: {
                return renderForm(<MIBParamsForm path={formPath} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.BLO: {
                return renderForm(<BLOParamsForm path={formPath} onSubmit={createDataset} onCancel={onCancel} />);
            }
        }
    }

}

export default connect(mapStateToProps, mapDispatchToProps)(DatasetOpen)