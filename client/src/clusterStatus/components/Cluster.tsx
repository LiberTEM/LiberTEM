import * as React from "react";
import { useEffect, useState } from "react";
import { connect } from "react-redux";
import { Button, Modal, Popup } from "semantic-ui-react";
import { ChannelStatusCodes } from "../../channel/reducers";
import { ClusterTypes } from "../../messages";
import { RootReducer } from "../../store";
import LocalStatus from "./LocalStatus";
import NotConnected from "./NotConnected";
import TCPStatus from "./TCPStatus";

const mapStateToProps = (state: RootReducer) => ({
    clusterConnection: state.clusterConnection,
    channelStatus: state.channelStatus.status,
    type: state.config.lastConnection.type,
    localcore: state.config.localCores,
    cudas: state.config.lastConnection.cudas,
    address: state.config.lastConnection.address,
})

type MergedProps = ReturnType<typeof mapStateToProps>;

const ClusterStatus: React.FC<MergedProps> = ({ clusterConnection, channelStatus, type, localcore, cudas, address }) => {
    enum ColorType {
        blue = "blue",
        grey = "grey",
        orange = "orange",
        red = "red",
    }

    enum IconType {
        plug = "plug",
        broken = "broken chain",
        wait = "refresh",
    }

    const [color, setColor] = useState<ColorType>(ColorType.grey)
    const [icon, setIcon] = useState<IconType>(IconType.plug)
    const [status, setStatus] = useState(clusterConnection.status)
    const [disable, setDisable] = useState(true)

    useEffect(() => {
        if (channelStatus === ChannelStatusCodes.CONNECTED 
            || channelStatus === ChannelStatusCodes.READY
            || channelStatus === ChannelStatusCodes.SNOOZED
            || channelStatus === ChannelStatusCodes.UNSNOOZING) {
            setStatus(clusterConnection.status)
            setDisable(false)
            if (channelStatus === ChannelStatusCodes.READY) {
                setColor(ColorType.blue)
                setStatus("connected")
                setIcon(IconType.plug)
            }
            else if (channelStatus ===ChannelStatusCodes.SNOOZED) {
                setColor(ColorType.grey)
                setStatus("snoozed")
                setIcon(IconType.broken)
            }
            else if (channelStatus === ChannelStatusCodes.UNSNOOZING) {
                setColor(ColorType.orange)
                setStatus("unsnoozing")
                setIcon(IconType.wait)
            } else {
                setColor(ColorType.grey)
                setStatus("unknown")
                setIcon(IconType.broken)
            }
        } else {
            setDisable(true)
            setStatus("unknown")
            setColor(ColorType.grey)
        }

    }, [clusterConnection, channelStatus, ColorType])


    const clusterDetails = () => {
        if (clusterConnection.status === "connected" ||
            clusterConnection.status === "snoozed" ||
            clusterConnection.status === "unsnoozing") {
            if (type === ClusterTypes.LOCAL) {
                return <LocalStatus cudas={cudas} localCores={localcore} />;
            } else {
                return <TCPStatus address={address} />;
            }
        } else {
            return <NotConnected />;
        }
    };

    return (
        <Modal
            trigger={
                <Button
                    color={color}
                    content="Cluster info"
                    icon={icon}
                    labelPosition="left"
                    disabled={disable}
                    label={{ as: "a", basic: true, content: status }} />}
            size="small" >
            <Popup.Header>Connection Info</Popup.Header>
            <Popup.Content>{clusterDetails()}</Popup.Content>
        </Modal>
    );
};

export default connect(mapStateToProps)(ClusterStatus);
