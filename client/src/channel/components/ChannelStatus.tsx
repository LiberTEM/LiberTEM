import * as React from "react";
import { connect } from "react-redux";
import { RootReducer } from "../../store";
import ChannelConnecting from "./ChannelConnecting";

const mapStateToProps = (state: RootReducer) => {
    return {
        channelStatus: state.channelStatus,
    }
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const messages = {
    waiting: "Waiting...",
    connecting: "Connecting...",
    connected: "Connected, waiting for initial state...",
}

const ChannelStatus: React.SFC<MergedProps> = ({ children, channelStatus }) => {
    switch (channelStatus.status) {
        case "waiting":
        case "connecting":
        case "connected": {
            return <ChannelConnecting msg={messages[channelStatus.status]} />;
        }
    }
    return <>{children}</>;
}

export default connect(mapStateToProps)(ChannelStatus);