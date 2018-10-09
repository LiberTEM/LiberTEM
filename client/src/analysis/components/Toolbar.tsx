import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Button, Icon, IconProps, Segment } from "semantic-ui-react";
import { JobReducerState } from "../../job/reducers";
import { RootReducer } from "../../store";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";

interface ToolbarProps {
    analysis: AnalysisState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: ToolbarProps) => {
    return {
        handleApply: () => dispatch(analysisActions.Actions.run(ownProps.analysis.id, "RESULT")),
        handleRemove: () => dispatch(analysisActions.Actions.remove(ownProps.analysis.id)),
    }
}

type MergedProps = ToolbarProps & ReturnType<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

const Toolbar: React.SFC<MergedProps> = ({ status, analysis, handleApply, handleRemove }) => {
    const running = status === "busy";
    const applyIconProps: IconProps = running ? { name: 'cog', loading: true } : { name: 'check' }
    return (
        <Segment attached="bottom">
            <Button.Group>
                <Button primary={true} onClick={handleApply} icon={true}>
                    <Icon {...applyIconProps} />
                    Apply
                </Button>
                <Button onClick={handleRemove} icon={true}>
                    <Icon name='remove' />
                    Remove
                </Button>
            </Button.Group>
        </Segment>
    );
}

const getAnalysisStatus = (analysis: AnalysisState, jobs: JobReducerState) => {
    const jobId = analysis.jobs.RESULT;
    if (jobId === undefined) {
        return "idle";
    }
    const isRunning = jobs.byId[jobId].running !== "DONE";
    return isRunning ? "busy" : "idle";
}

const mapStateToProps = (state: RootReducer, ownProps: ToolbarProps) => {
    return {
        status: getAnalysisStatus(ownProps.analysis, state.jobs),
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(Toolbar);