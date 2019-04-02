import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Button, Icon, IconProps, Segment } from "semantic-ui-react";
import { JobReducerState } from "../../job/reducers";
import { JobRunning } from "../../job/types";
import { RootReducer } from "../../store";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";

interface ToolbarProps {
    analysis: AnalysisState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: ToolbarProps) => {
    return {
        handleApply: () => {
            dispatch(analysisActions.Actions.run(ownProps.analysis.id, "RESULT"))
        },
        handleRemove: () => dispatch(analysisActions.Actions.remove(ownProps.analysis.id)),
    }
}

type MergedProps = ToolbarProps & ReturnType<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

const Toolbar: React.SFC<MergedProps> = ({ status, handleApply, handleRemove }) => {
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

const getAnalysisStatus = (analysis: AnalysisState, jobs: JobReducerState): "idle" | "busy" => {
    const jobId = analysis.jobs.RESULT;
    if (jobId === undefined) {
        return "idle";
    }
    const isDone = jobs.byId[jobId].running === JobRunning.DONE;
    return isDone ? "idle" : "busy";
}

const mapStateToProps = (state: RootReducer, ownProps: ToolbarProps) => {
    const status = getAnalysisStatus(ownProps.analysis, state.jobs);
    return {
        status,
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(Toolbar);