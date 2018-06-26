import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Dropdown, DropdownItemProps, Header, Segment } from 'semantic-ui-react';
import * as analysisActions from '../../analysis/actions';
import AnalysisList from "../../analysis/components/AnalysisList";
import { Analysis, AnalysisMetadata, AnalysisTypes } from "../../analysis/types";
import { filterWithPred } from "../../helpers/reducerHelpers";
import { RootReducer } from "../../store";
import { Dataset } from "../types";

interface DatasetProps {
    dataset: Dataset
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: DatasetProps) => {
    return {
        createAnalysis: analysisActions.Actions.create,
        handleAddAnalysis: (_: any, data: DropdownItemProps) => {
            dispatch(analysisActions.Actions.create(ownProps.dataset.id, data.value as AnalysisTypes));
        },
    }
}

const mapStateToProps = (state: RootReducer, ownProps: DatasetProps) => {
    const p = (analysis: Analysis) => analysis.dataset === ownProps.dataset.id;
    return {
        analyses: filterWithPred(state.analyses, p),
    }
}

type MergedProps = DatasetProps & ReturnType<typeof mapStateToProps> & ReturnType<typeof mapDispatchToProps>;

const DatasetComponent: React.SFC<MergedProps> = ({ dataset, analyses, handleAddAnalysis }) => {
    const analysisTypeKeys = Object.keys(AnalysisTypes).filter(k => typeof AnalysisTypes[k as any] === "string");
    const analysisTypeOptions = analysisTypeKeys.map(t => ({
        text: AnalysisMetadata[AnalysisTypes[t as any]].short,
        value: AnalysisTypes[t as any],
    }));

    return (
        <Segment>
            <Header as="h2">{dataset.name}</Header>

            <Dropdown text='Add analysis' icon='add' floating={true} labeled={true} button={true} className='icon'>
                <Dropdown.Menu>
                    <Dropdown.Header content='implemented analyses' />
                    {analysisTypeOptions.map(option => <Dropdown.Item key={option.value} onClick={handleAddAnalysis} {...option} />)}
                </Dropdown.Menu>
            </Dropdown>

            <AnalysisList analyses={analyses} />
        </Segment>
    );
}

const DatasetContainer = connect(mapStateToProps, mapDispatchToProps)(DatasetComponent);

export default DatasetContainer;