import * as React from "react";
import { connect } from "react-redux";
import { Button, Icon } from "semantic-ui-react";
import { RootReducer } from "../../store";
import * as browserActions from '../actions';
import FileBrowser from "./FileBrowser";

export const mapStateToProps = (state: RootReducer) => {
    return {
        isOpen: state.browser.isOpen,
        busy: state.openDataset.busy,
        formVisible: state.openDataset.formVisible,
    }
}

export const mapDispatchToProps = {
    open: browserActions.Actions.open,
}

type MergedProps = ReturnType<typeof mapStateToProps> & DispatchProps<typeof mapDispatchToProps>;

const BrowserWrapper: React.SFC<MergedProps> = ({ formVisible, isOpen, open, busy }) => {
    if(formVisible || busy) {
        return null;
    } else if (!isOpen) {
        return (
            <Button icon={true} labelPosition="left" onClick={open}>
                <Icon name='add' />
                Browse
            </Button>
        );
    } else {
        return (
            <FileBrowser />
        );
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(BrowserWrapper)
