import * as React from "react";
import { useEffect, useState } from "react";
import { connect } from "react-redux";
import { Button, List, Header, Modal, Segment} from "semantic-ui-react";
import { RootReducer } from "../../store";
import { HostDetails } from "../../messages"
import { getClusterDetail } from "../api"

const ClusterDetails = () => {
    const initialHost: HostDetails[] = [
        {
            host: "",
            cpu: 0,
            cuda: 0,
            service: 0,
        },
    ];

    const [clusterDetails, setClusterDetails] = useState({
        status: "",
        messageType: "",
        details: initialHost,
    });

    useEffect(() => {
        const getData = async () => {
            await getClusterDetail().then(currentDetails => {
                setClusterDetails({
                    status: currentDetails.status,
                    messageType: currentDetails.messageType,
                    details: currentDetails.details,
                });
            });
        };
        getData();
    }, []);

    return clusterDetails.details.map((details: HostDetails) => {
        return (
            <Segment>
            <List.Item>
                <List.Content>host : {details.host}</List.Content>
                <List.Content>CPU : {details.cpu}</List.Content>
                <List.Content>CUDA : {details.cuda}</List.Content>
            </List.Item>
            </Segment>
        );
    });
};

const mapStateToProps = (state: RootReducer) => {
    return {
        address: state.config.lastConnection.address,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

const TCPStatus: React.SFC<MergedProps> = ({ address }) => {
    const template = [
        `import libertem.api as lt`,
        `import distributed as dd`,
        `from libertem.executor.dask import DaskJobExecutor\n`,
        `client = dd.Client("URI")`,
        `executor = DaskJobExecutor(client)\n`,
        `ctx = lt.Context(executor=executor)`,
    ];

    const connectionCode = template.join("\n");
    const code = connectionCode.replace("URI", address);
    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
    };

    return (
        <Modal.Content>
            <List>
                <Header as='h4'attached='top'>
                Connected to {address}
                </Header>
                <Segment.Group >
                {ClusterDetails()}
                </Segment.Group>
                <List.Item>
                    <List.Content>
                        <Segment.Group>
                            <Segment as="h5">Connection code</Segment>
                            <Segment>
                                <Button floated={"right"} icon={"copy"} onClick={copyToClipboard} />
                                {code.split("\n").map((item, i) => {
                                    return <p key={i}>{item}</p>;
                                })}
                            </Segment>
                        </Segment.Group>
                    </List.Content>
                </List.Item>
            </List>
        </Modal.Content>
    );
};

export default connect(mapStateToProps)(TCPStatus);
