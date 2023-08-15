import * as React from "react";
import { useSelector } from "react-redux";
import ClusterConnectionForm from "../../cluster/components/ClusterConnectionForm";
import { assertNotReached } from "../../helpers";
import { RootReducer } from "../../store";
import ChannelConnecting from "./ChannelConnecting";
import ChannelShutdown from './ChannelShutdown'

const messages = {
    waiting: "Waiting...",
    connecting: "Connecting...",
}

const clusterMessages = {
    connected: "Connected, waiting for initial state...",
    unknown: "Connected, fetching cluster status...",
    connecting: "Connecting to cluster"
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
    assertNotReached("should not happen");
    return null;
}

const ChannelStatus: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
    const channelStatus = useSelector((state: RootReducer) => state.channelStatus);

    switch (channelStatus.status) {
        case "waiting":
        case "connecting": {
            return <ChannelConnecting msg={messages[channelStatus.status]} />;
        }
        case "connected": {
            return <ConnectedNotReady />
        }
        case "ready":
            return <>{children}</>;
        case "disconnected":
            return <ChannelShutdown />
        default:
            assertNotReached("should not happen");
            return null;
    }
}

export default ChannelStatus;