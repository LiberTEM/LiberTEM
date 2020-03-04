import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Button, Icon, IconProps, Segment } from "semantic-ui-react";
import { RootReducer } from "../../store";
import * as analysisActions from "../actions";
import { getAnalysisStatus } from "../helpers";
import { CompoundAnalysisState } from "../types";

interface ToolbarProps {
    compoundAnalysis: CompoundAnalysisState,
    busyIdxs: number[],
    onApply: () => void,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: ToolbarProps) => {
    return {
        handleRemove: () => dispatch(analysisActions.Actions.remove(ownProps.compoundAnalysis.id)),
    }
}

const mapStateToProps = (state: RootReducer, ownProps: ToolbarProps) => {
    const status = getAnalysisStatus(
        ownProps.compoundAnalysis, state.analyses, state.jobs,
        ownProps.busyIdxs
    );
    return {
        status,
    };
}

type MergedProps = ToolbarProps & ReturnType<typeof mapDispatchToProps> & ReturnType<typeof mapStateToProps>;

const Toolbar: React.SFC<MergedProps> = ({ status, onApply, handleRemove }) => {
    const running = status === "busy";
    const applyIconProps: IconProps = running ? { name: 'cog', loading: true } : { name: 'check' }
    return (
        <Segment attached="bottom">
            <Button.Group>
                <Button primary={true} onClick={onApply} icon={true}>
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

export default connect(mapStateToProps, mapDispatchToProps)(Toolbar);