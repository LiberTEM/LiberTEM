import * as React from 'react';
import { Icon, Message } from 'semantic-ui-react';

interface ChannelConnectingProps {
    msg: string,
}

const ChannelConnecting: React.FC<ChannelConnectingProps> = ({ msg }) => (
    <Message icon>
        <Icon name='cog' loading />
        <Message.Content>
            <Message.Header>Connecting to LiberTEM</Message.Header>
            {msg}
        </Message.Content>
    </Message>
)

export default ChannelConnecting;