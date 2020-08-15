import * as React from "react";
import { connect } from "react-redux";
import { List, Modal } from "semantic-ui-react";
import { RootReducer } from "../../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        localCore: state.config.localCores,
        numWorker: state.config.lastConnection.numWorker,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

const LocalStatus: React.SFC<MergedProps> = ({ localCore, numWorker }) => {
    return (
        <Modal.Content>
            <List>
                <List.Item>
                    <List.Content as="h4">Connected to local cluster</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Local core : {localCore}</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>CPU Workers : {numWorker} </List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>CUDA : </List.Content>
                </List.Item>
            </List>
        </Modal.Content>
    );
};

export default connect(mapStateToProps)(LocalStatus);
