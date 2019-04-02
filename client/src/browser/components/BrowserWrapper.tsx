import * as React from "react";
import { connect } from "react-redux";
import { Button, Icon } from "semantic-ui-react";
import { RootReducer } from "../../store";
import * as browserActions from '../actions';
import FileBrowser from "./FileBrowser";

export const mapStateToProps = (state: RootReducer) => {
    return {
        isOpen: state.browser.isOpen,
    }
}

export const mapDispatchToProps = {
    open: browserActions.Actions.open,
}

type MergedProps = ReturnType<typeof mapStateToProps> & DispatchProps<typeof mapDispatchToProps>;

const BrowserWrapper: React.SFC<MergedProps> = ({ isOpen, open }) => {
    if (!isOpen) {
        return (
            <Button icon={true} labelPosition="left" onClick={open}>
                <Icon name='add' />
                Browse
            </Button>
        );
    }
    return (
        <FileBrowser />
    );
}

export default connect(mapStateToProps, mapDispatchToProps)(BrowserWrapper)