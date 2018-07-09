import * as React from "react";
import { connect, Dispatch } from 'react-redux';
import { Segment } from "semantic-ui-react";
import * as uuid from "uuid/v4";
import { DatasetFormParams, DatasetTypes } from '../../messages';
import * as datasetActions from "../actions";
import HDF5ParamsForm from "./HDF5ParamsForm";
import HDFSParamsForm from './HDFSParamsForm';
import RawFileParamsForm from "./RawFileParamsForm";

const mapDispatchToProps = (dispatch: Dispatch, ownProps: ExtraProps) => {
    return {
        createDataset: (params: DatasetFormParams) => {
            dispatch(datasetActions.Actions.create({
                id: uuid(),
                params,
            }));
            ownProps.onSubmit();
        },
    };
}

interface ExtraProps {
    type: DatasetTypes,
    onSubmit: () => void,
    onCancel: () => void,
}

type MergedProps = ExtraProps & ReturnType<typeof mapDispatchToProps>;

const DatasetOpen: React.SFC<MergedProps> = ({ type, createDataset, onCancel }) => {
    const renderForm = (form: React.ReactNode) => {
        return (
            <Segment>
                {form}
            </Segment>
        );
    }
    switch (type) {
        case DatasetTypes.HDF5: {
            return renderForm(<HDF5ParamsForm onSubmit={createDataset} onCancel={onCancel} />);
        }
        case DatasetTypes.HDFS: {
            return renderForm(<HDFSParamsForm onSubmit={createDataset} onCancel={onCancel} />);
        }
        case DatasetTypes.RAW: {
            return renderForm(<RawFileParamsForm onSubmit={createDataset} onCancel={onCancel} />);
        }
    }
}

export default connect<{}, {}, ExtraProps>(null, mapDispatchToProps)(DatasetOpen)