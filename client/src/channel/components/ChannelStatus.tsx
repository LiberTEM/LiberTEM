import * as React from "react";
import { connect } from "react-redux";
import ClusterConnectionForm from "../../cluster/components/ClusterConnectionForm";
import { RootReducer } from "../../store";
import ChannelConnecting from "./ChannelConnecting";

const mapStateToProps = (state: RootReducer) => {
    return {
        channelStatus: state.channelStatus,
        clusterConnection: state.clusterConnection,
        haveConfig: state.config.haveConfig,
    }
}


type MergedProps = ReturnType<typeof mapStateToProps>;

const messages = {
    waiting: "Waiting...",
    connecting: "Connecting...",
}

const clusterMessages = {
    connected: "Connected, waiting for initial state...",
    unknown: "Connected, fetching cluster status...",
}

const ChannelStatus: React.SFC<MergedProps> = ({ haveConfig, children, channelStatus, clusterConnection }) => {
    switch (channelStatus.status) {
        case "waiting":
        case "connecting": {
            return <ChannelConnecting msg={messages[channelStatus.status]} />;
        }
        case "connected": {
            if (!haveConfig) {
                return <ChannelConnecting msg="waiting for configuration..." />;
            }
            if (clusterConnection.status === "disconnected") {
                return <ClusterConnectionForm />
            } else if (clusterConnection.status === "connected") {
                return <ChannelConnecting msg={clusterMessages.connected} />;
            } else if (clusterConnection.status === "unknown") {
                return <ChannelConnecting msg={clusterMessages.unknown} />;
            }
        }
    }
    return <>{children}</>;
}

export default connect(mapStateToProps)(ChannelStatus);