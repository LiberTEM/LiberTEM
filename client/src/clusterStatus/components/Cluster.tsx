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
    enum ColorType  {
        blue= "blue",
        grey= "grey"
    }

    const [color , setColor] = useState<ColorType>(ColorType.grey)
    const [status, setStatus] =  useState(clusterConnection.status)
    const [disable, setDisable] = useState(true)

    useEffect(()=>{
        if (channelStatus === ChannelStatusCodes.CONNECTED || channelStatus === ChannelStatusCodes.READY) {
            setStatus(clusterConnection.status)
            setDisable(false)
            if (clusterConnection.status === "connected") {
                setColor(ColorType.blue)
            }else{
                setColor(ColorType.grey)
            }
        } else {
            setDisable(true)
            setStatus("unknown")
            setColor(ColorType.grey)
        }

    }, [clusterConnection, channelStatus, ColorType])


    const clusterDetails = () => {
        if (clusterConnection.status === "connected") {
            if (type ===ClusterTypes.LOCAL) {
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
                    color= {color}
                    content="Cluster info"
                    icon="plug"
                    labelPosition="left"
                    disabled={disable}
                    label={{ as: "a", basic: true, content: status }} /> }
            size="small" >
            <Popup.Header>Connection Info</Popup.Header>
            <Popup.Content>{clusterDetails()}</Popup.Content>
        </Modal>
    );
};

export default connect(mapStateToProps)(ClusterStatus);
