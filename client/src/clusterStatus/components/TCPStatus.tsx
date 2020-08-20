import * as React from "react";
import { useEffect, useState } from "react";
import { connect } from "react-redux";
import { Accordion, Button, Header, Icon, List, Modal, Segment } from "semantic-ui-react";
import { HostDetails } from "../../messages";
import { RootReducer } from "../../store";
import { getClusterDetail } from "../api";

const ClusterDetails = () => {
    const [clustOverview, setOverview]= useState({
            host: 0,
            cpu: 0,
            cuda: 0,
    })
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

    const [idx, setIdx] = useState(false);
    const [expandMsg, setMsg] = useState("More Info");

    const handleClick = () => {
        setIdx(!idx);
        if (expandMsg === "More Info") {
            setMsg("Less Info");
        } else {
            setMsg("More Info");
        }
    };

    useEffect(() => {
        const getData = async () => {
            await getClusterDetail().then(currentDetails => {
                setClusterDetails({
                    status: currentDetails.status,
                    messageType: currentDetails.messageType,
                    details: currentDetails.details,
                });
                const overview = {
                    host: 0,
                    cpu: 0,
                    cuda: 0,
                }
                currentDetails.details.forEach((node)=>{
                    overview.host+=1
                    overview.cpu+=node.cpu
                    overview.cuda+=node.cuda
                })
                setOverview(overview)
            });
        };
        getData();
    }, []);

    const clusterExpanded = clusterDetails.details.map((details: HostDetails) => {
        return (
            <Segment>
                <List.Item  key={details.host}>
                    <List.Content>host : {details.host}</List.Content>
                    <List.Content>CPU : {details.cpu}</List.Content>
                    <List.Content>CUDA : {details.cuda}</List.Content>
                </List.Item>
            </Segment>
        );
    });

    return (
        <>
            <Segment>
                <List.Item>
                    <List.Content>host : {clustOverview.host}</List.Content>
                    <List.Content>CPU : {clustOverview.cpu}</List.Content>
                    <List.Content>CUDA : {clustOverview.cuda}</List.Content>
                </List.Item>
            </Segment>
            <Accordion>
                <Accordion.Title active={idx} onClick={handleClick}>
                    <Icon name="dropdown" />
                    {expandMsg}
                </Accordion.Title>
                <Accordion.Content active={idx}>
                    <Segment.Group>{clusterExpanded}</Segment.Group>
                </Accordion.Content>
            </Accordion>
        </>
    );
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
