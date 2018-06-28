import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Header } from 'semantic-ui-react';
import * as analysisActions from '../../analysis/actions';
import AnalysisList from "../../analysis/components/AnalysisList";
import AnalysisSelect from "../../analysis/components/AnalysisSelect";
import { AnalysisState, AnalysisTypes } from "../../analysis/types";
import { filterWithPred } from "../../helpers/reducerHelpers";
import { DatasetState } from "../../messages";
import { RootReducer } from "../../store";

interface DatasetProps {
    dataset: DatasetState
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: DatasetProps) => {
    return {
        createAnalysis: analysisActions.Actions.create,
        handleAddAnalysis: (type: AnalysisTypes) => {
            dispatch(analysisActions.Actions.create(ownProps.dataset.id, type));
        },
    }
}

const mapStateToProps = (state: RootReducer, ownProps: DatasetProps) => {
    const p = (analysis: AnalysisState) => analysis.dataset === ownProps.dataset.id;
    return {
        analyses: filterWithPred(state.analyses, p),
    }
}

type MergedProps = DatasetProps & ReturnType<typeof mapStateToProps> & ReturnType<typeof mapDispatchToProps>;

const DatasetComponent: React.SFC<MergedProps> = ({ dataset, analyses, handleAddAnalysis }) => {
    return (
        <>
            <Header as="h2" dividing={true}>{dataset.name}</Header>
            <AnalysisSelect onClick={handleAddAnalysis} label='Add analysis' />
            <AnalysisList analyses={analyses} />
        </>
    );
}

const DatasetContainer = connect(mapStateToProps, mapDispatchToProps)(DatasetComponent);

export default DatasetContainer;