import * as React from "react";
import { connect } from "react-redux";
import { Button, Modal, Popup } from "semantic-ui-react";
import { RootReducer } from "../store";
import LocalStatus from "./localStatus";
import NotConnected from "./NotConnected";
import TCPStatus from "./TCPStatus";

const mapStateToProps = (state: RootReducer) => {
    return {
        clusterConnection: state.clusterConnection,
        type: state.config.lastConnection.type,
        localcore: state.config.localCores,
        cudas: state.config.lastConnection.cudas,
        address: state.config.lastConnection.address,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

const ClusterStatus: React.SFC<MergedProps> = ({ clusterConnection, type, localcore, cudas, address }) => {
    const clusterDetails = () => {
        if (clusterConnection.status === "connected") {
            const { details } = clusterConnection;
            if (type === "LOCAL") {
                return <LocalStatus cudas={cudas} details={details} localCore={localcore} />;
            } else {
                return <TCPStatus address={address} details={details} />;
            }
        } else {
            return <NotConnected />;
        }
    };

    return (
        <Modal trigger={<Button content={clusterConnection.status} />} size="small">
            <Popup.Header>Connection Info</Popup.Header>
            <Popup.Content>{clusterDetails()}</Popup.Content>
        </Modal>
    );
};

export default connect(mapStateToProps)(ClusterStatus);
