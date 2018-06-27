import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { Button, Icon, Segment } from "semantic-ui-react";
import * as analysisActions from "../actions";
import { Analysis } from "../types";

interface ToolbarProps {
    analysis: Analysis,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: ToolbarProps) => {
    return {
        handleApply: () => dispatch(analysisActions.Actions.run(ownProps.analysis.id)),
        handleRemove: () => dispatch(analysisActions.Actions.remove(ownProps.analysis.id)),
    }
}

type MergedProps = ToolbarProps & ReturnType<typeof mapDispatchToProps>;

const Toolbar: React.SFC<MergedProps> = ({ analysis, handleApply, handleRemove }) => {
    return (
        <Segment attached="bottom">
            <Button.Group>
                <Button primary={true} onClick={handleApply} icon={true}>
                    <Icon name='check' />
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

export default connect(null, mapDispatchToProps)(Toolbar);