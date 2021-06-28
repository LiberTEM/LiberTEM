import * as React from "react";
import { Icon, Message } from "semantic-ui-react";

const ChannelShutdown: React.FC = () => (
    <Message negative icon>
        <Icon name="shutdown" />
        <Message.Content>
            <Message.Header>Connection is closed</Message.Header>
            <p>please close the tab</p>
        </Message.Content>
    </Message>
);

export default ChannelShutdown;
