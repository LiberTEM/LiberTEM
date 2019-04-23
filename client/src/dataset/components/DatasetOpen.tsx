import * as React from "react";
import { connect } from 'react-redux';
import { Dispatch } from "redux";
import { DropdownProps, Header, Segment } from "semantic-ui-react";
import uuid from "uuid/v4";
import { assertNotReached } from "../../helpers";
import { DatasetFormParams, DatasetTypes } from '../../messages';
import { RootReducer } from "../../store";
import * as datasetActions from "../actions";
import BLOParamsForm from "./BLOParamsForm";
import DatasetTypeSelect from "./DatasetTypeSelect";
import EMPADParamsForm from "./EMPADParamsForm";
import FRMS6ParamsForm from "./FRMS6ParamsForm";
import HDF5ParamsForm from "./HDF5ParamsForm";
import HDFSParamsForm from './HDFSParamsForm';
import K2ISParamsForm from "./K2ISParamsForm";
import MIBParamsForm from "./MIBParamsForm";
import RawFileParamsForm from "./RawFileParamsForm";
import SERParamsForm from "./SERParamsForm";

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
        formInitial: state.openDataset.formInitialParams,
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

    constructor(props: MergedProps) {
        super(props);
        if (props.formInitial !== undefined) {
            this.state = {
                datasetType: props.formInitial.type,
            };
        }
    }

    public setDatasetType = (e: React.SyntheticEvent, data: DropdownProps) => {
        const type = data.value as DatasetTypes;
        this.setState({
            datasetType: type,
        })
    }

    public render() {
        const { formPath, formInitial, createDataset, onCancel } = this.props;
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
        switch (datasetType) {
            case DatasetTypes.HDF5: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<HDF5ParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.HDFS: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<HDFSParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.RAW: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<RawFileParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.MIB: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<MIBParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.BLO: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<BLOParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.K2IS: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<K2ISParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.SER: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<SERParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.FRMS6: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<FRMS6ParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />);
            }
            case DatasetTypes.EMPAD: {
                const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
                return renderForm(<EMPADParamsForm path={formPath} initial={initial} onSubmit={createDataset} onCancel={onCancel} />)
            }
        }
        return assertNotReached("unknown dataset type");
    }

}

export default connect(mapStateToProps, mapDispatchToProps)(DatasetOpen)