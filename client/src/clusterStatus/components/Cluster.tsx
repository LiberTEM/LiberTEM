import * as React from "react";
import { connect } from "react-redux";
import { Button, Modal, Popup } from "semantic-ui-react";
import { RootReducer } from "../../store";
import LocalStatus from "./localStatus";
import NotConnected from "./NotConnected";
import TCPStatus from "./TCPStatus";

const mapStateToProps = (state: RootReducer) => {
    return {
        connection: state.clusterConnection.status,
        type: state.config.lastConnection.type,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

const ClusterStatus: React.SFC<MergedProps> = ({ type, connection }) => {
    const clusterDetails = () => {
        if (connection === "connected" && type === "LOCAL") {
            return <LocalStatus />;
        } else if (connection === "connected" && type === "TCP") {
            return <TCPStatus />;
        } else {
            return <NotConnected />;
        }
    };

    return (
        <Modal trigger={<Button content={connection} />} size="small">
            <Popup.Header>Connection Info</Popup.Header>
            <Popup.Content>{clusterDetails()}</Popup.Content>
        </Modal>
    );
};

export default connect(mapStateToProps)(ClusterStatus);
