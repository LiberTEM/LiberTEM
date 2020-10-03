import * as React from "react";
import { useEffect, useState } from "react";
import { Accordion, Button, Header, Icon, List, Modal, Segment } from "semantic-ui-react";
import { HostDetails } from "../../messages";
import { getClusterDetail } from "../api";


const ClusterDetails = (details: HostDetails[]) => {
    const [clustOverview, setOverview] = useState({
        host: 0,
        cpu: 0,
        cuda: 0,
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
        const overview = {
            host: 0,
            cpu: 0,
            cuda: 0,
        };
        details.forEach(node => {
            overview.host += 1;
            overview.cpu += node.cpu;
            overview.cuda += node.cuda;
        });
        setOverview(overview);
    }, [details]);

    const clusterExpanded = details.map((node: HostDetails) => {
        return (
            <Segment key={node.host}>
                <List.Item >
                    <List.Content>Host : {node.host}</List.Content>
                    <List.Content>Number of CPU workers : {node.cpu}</List.Content>
                    <List.Content>Number of CUDA workers : {node.cuda}</List.Content>
                </List.Item>
            </Segment>
        );
    });

    return (
        <>
            <Segment>
                <List.Item>
                    <List.Content>Number of hosts : {clustOverview.host}</List.Content>
                    <List.Content>Number of CPU workers : {clustOverview.cpu}</List.Content>
                    <List.Content>Number of CUDA workers : {clustOverview.cuda}</List.Content>
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

interface TCPStatusProps {
    address: string;
}

const TCPStatus: React.SFC<TCPStatusProps> = ({ address }) => {
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

    const [clustDetails, setDetails] = useState<HostDetails[]>([])

    useEffect(() => {
        const updateDetails = async () => {
            await getClusterDetail().then(newDetails => {
                setDetails(newDetails.details)
            })}
        updateDetails()
        }, [])

    return (
        <Modal.Content>
            <List>
                <Header as="h4" attached="top">
                    Connected to {address}
                </Header>
                <Segment.Group>{ClusterDetails(clustDetails)}</Segment.Group>
                <List.Item>
                    <List.Content>
                        <Segment.Group>
                            <Segment as="h5">Connection code</Segment>
                            <Segment>
                                <Button floated={"right"} icon={"copy"} onClick={copyToClipboard} />
                                <pre>
                                    <code>
                                        {code}
                                    </code>
                                </pre>
                            </Segment>
                        </Segment.Group>
                    </List.Content>
                </List.Item>
            </List>
        </Modal.Content>
    );
};

export default TCPStatus;
