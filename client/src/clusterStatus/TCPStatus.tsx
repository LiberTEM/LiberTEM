import * as React from "react";
import { connect } from "react-redux";
import { Button, List, Modal, Segment } from "semantic-ui-react";
import { RootReducer } from "../store";

const mapStateToProps = (state: RootReducer) => {
    return {
        address: state.config.lastConnection.address,
    };
};

type MergedProps = ReturnType<typeof mapStateToProps>;

const TCPStatus: React.SFC<MergedProps> = ({ address }) => {
    const template = [
                `from libertem.api import lt`, 
                `import distributed as dd\n`, 
                `client = dd.Client("URI")`, 
                `ctx = lt.Context(executor=client)`
            ];

    const connectionCode = template.join("\n");
    const code = connectionCode.replace("URI", address);
    const copyToClipboard = () => {
        navigator.clipboard.writeText(code);
    };

    return (
        <Modal.Content>
            <List>
                <List.Item>
                    <List.Content as="h4">Connected to {address}</List.Content>
                </List.Item>
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
