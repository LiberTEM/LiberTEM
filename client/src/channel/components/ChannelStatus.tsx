import * as React from "react";
import { useSelector } from "react-redux";
import ClusterConnectionForm from "../../cluster/components/ClusterConnectionForm";
import { assertNotReached } from "../../helpers";
import { RootReducer } from "../../store";
import { ChannelStatusCodes } from "../reducers";
import ChannelConnecting from "./ChannelConnecting";
import ChannelShutdown from './ChannelShutdown';

const messages = {
    waiting: "Waiting...",
    connecting: "Connecting...",
}

const clusterMessages = {
    connected: "Connected, waiting for initial state...",
    unknown: "Connected, fetching cluster status...",
    connecting: "Connecting to cluster",
    snoozed: "Cluster is snoozed",
    unsnoozing: "Cluster is unsnoozing"
}

const ConnectedNotReady: React.FC = () => {
    const haveConfig = useSelector((state: RootReducer) => state.config.haveConfig);
    const clusterConnection = useSelector((state: RootReducer) => state.clusterConnection);

    if (!haveConfig) {
        return <ChannelConnecting msg="waiting for configuration..." />;
    }
    if (clusterConnection.status === "disconnected") {
        return <ClusterConnectionForm />
    } else if (clusterConnection.status === "connected") {
        return <ChannelConnecting msg={clusterMessages.connected} />;
    } else if (clusterConnection.status === "unknown") {
        return <ChannelConnecting msg={clusterMessages.unknown} />;
    } else if (clusterConnection.status === "connecting") {
        return <ChannelConnecting msg={clusterMessages.connecting} />
    }
    // else if (clusterConnection.status === "snoozed") {
        // return <ChannelConnecting msg={clusterMessages.snoozed} />
    // } else if (clusterConnection.status === "unsnoozing") {
        // return <ChannelConnecting msg={clusterMessages.unsnoozing} />
    // }
    assertNotReached("should not happen");
    return null;
}

const ChannelStatus: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
    const channelStatus = useSelector((state: RootReducer) => state.channelStatus);

    switch (channelStatus.status) {
        case ChannelStatusCodes.WAITING:
        case ChannelStatusCodes.CONNECTING: {
            return <ChannelConnecting msg={messages[channelStatus.status]} />;
        }
        case ChannelStatusCodes.CONNECTED: {
            return <ConnectedNotReady />
        }
        case ChannelStatusCodes.READY:
        case ChannelStatusCodes.SNOOZED:
        case ChannelStatusCodes.UNSNOOZING:
            return <>{children}</>;
        case ChannelStatusCodes.DISCONNECTED:
            return <ChannelShutdown />
        default:
            assertNotReached("should not happen");
            return null;
    }
}

export default ChannelStatus;