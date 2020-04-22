import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import * as compoundAnalysisActions from "../../compoundAnalysis/actions";
import AnalysisSelect from "../../compoundAnalysis/components/base/AnalysisSelect";
import { AnalysisTypes, DatasetState } from "../../messages";

interface DatasetProps {
    dataset: DatasetState
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: DatasetProps) => {
    return {
        handleAddAnalysis: (type: AnalysisTypes) => {
            dispatch(compoundAnalysisActions.Actions.create(ownProps.dataset.id, type));
        },
    }
}
type MergedProps = DatasetProps & ReturnType<typeof mapDispatchToProps>;

const AddAnalysis: React.SFC<MergedProps> = ({ handleAddAnalysis }) => {
    return <AnalysisSelect onClick={handleAddAnalysis} label='Add analysis' />
}


export default connect(null, mapDispatchToProps)(AddAnalysis);