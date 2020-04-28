import * as React from "react";
import { connect } from "react-redux";
import { Button, Icon } from "semantic-ui-react";
import * as datasetActions from '../../dataset/actions';
import { RootReducer } from "../../store";
import * as browserActions from '../actions';
import BottomComponent from "../empty_state/BottomComponent";
import TopComponent from "../empty_state/TopComponent";
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
    detect: datasetActions.Actions.detect,
}


type MergedProps = ReturnType<typeof mapStateToProps> & DispatchProps<typeof mapDispatchToProps>;

const BrowserWrapper: React.SFC<MergedProps> = ({ formVisible, isOpen, open, busy }) => {
    if(formVisible || busy) {
        return null;
    } else if (!isOpen) {
        return (
                <div style={{textAlign: 'center'}}>
                    <TopComponent />
                    <Button icon={true} labelPosition="left" onClick={open} color='blue'>
                        <Icon name='add' />
                        Browse
                    </Button>
                    <BottomComponent />
                </div>
        );
    } else {
        return (
            <FileBrowser />
        );
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(BrowserWrapper)
