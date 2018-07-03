import * as React from "react";
import { connect } from "react-redux";
import { RootReducer } from "../../store";
import ChannelConnecting from "./ChannelConnecting";
import ClusterConnectionForm from "./ClusterConnectionForm";

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
        case "connecting": {
            return <ChannelConnecting msg={messages[channelStatus.status]} />;
        }
        case "connected": {
            // tslint:disable:no-console
            // tslint:disable:jsx-no-lambda
            return <ClusterConnectionForm onSubmit={(e) => console.log(e)} />
        }
    }
    return <>{children}</>;
}

export default connect(mapStateToProps)(ChannelStatus);