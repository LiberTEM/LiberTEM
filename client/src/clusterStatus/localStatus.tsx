import * as React from "react";
import { connect } from "react-redux";
import { List, Modal } from "semantic-ui-react";
import { RootReducer } from "../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        localCore: state.config.localCores,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

const LocalStatus: React.SFC<MergedProps> = ({ localCore }) => {
    return (
        <Modal.Content>
            <List>
                <List.Item>
                    <List.Content as="h4">Connected to local cluster</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>{localCore} Local core</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>X CPU Workers</List.Content>
                </List.Item>
                <List.Item>
                    <List.Content>Y GPUs</List.Content>
                </List.Item>
            </List>
        </Modal.Content>
    );
};

export default connect(mapStateToProps)(LocalStatus);
