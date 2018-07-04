import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Grid, Header, Segment } from "semantic-ui-react";
import { defaultDebounce } from "../../helpers";
import JobComponent from "../../job/components/Job";
import { CenterOfMassParams, DatasetState } from "../../messages";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    parameters: CenterOfMassParams,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }));
        }),
        handleRChange: defaultDebounce((r: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { r }));
        }),
    }
}


type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const CenterOfMassAnalysis: React.SFC<MergedProps> = ({ parameters, analysis, dataset, handleRChange, handleCenterChange }) => {
    const { currentJob } = analysis;
    const { shape } = dataset.params;

    const resultWidth = shape[1];
    const resultHeight = shape[0];

    // TODO: center of mass needs the reference center as parameter
    return (
        <>
            <Header as='h3' attached="top">COM analysis</Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            &nbsp;
                        </Grid.Column>
                        <Grid.Column>
                            <JobComponent job={currentJob} width={resultWidth} height={resultHeight} />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(CenterOfMassAnalysis);