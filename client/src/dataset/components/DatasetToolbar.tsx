import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Button } from "semantic-ui-react";
import { DatasetState } from "../../messages";
import * as datasetActions from "../actions";

interface DatasetProps {
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: DatasetProps) => {
    return {
        handleRemoveDataset: () => {
            dispatch(datasetActions.Actions.delete(ownProps.dataset.id));
        }
    }
}

type MergedProps = DatasetProps & ReturnType<typeof mapDispatchToProps>;

const DatasetToolbar: React.SFC<MergedProps> = ({ dataset, handleRemoveDataset }) => {
    return (
        <>
            <Button icon="remove" labelPosition="left" onClick={handleRemoveDataset} content='Close Dataset' />
        </>
    );
}


export default connect(null, mapDispatchToProps)(DatasetToolbar);