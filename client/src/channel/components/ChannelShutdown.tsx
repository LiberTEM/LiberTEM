import * as React from "react";
import { Icon, Message } from "semantic-ui-react";

const ChannelShutdown: React.SFC = () => (
    <Message negative={true} icon={true}>
        <Icon name="shutdown" />
        <Message.Content>
            <Message.Header>Connection is closed</Message.Header>
            <p>please close the tab</p>
        </Message.Content>
    </Message>
);

export default ChannelShutdown;
