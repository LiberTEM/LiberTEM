import * as React from 'react';
import { Icon, Message } from 'semantic-ui-react';

interface ChannelConnectingProps {
    msg: string,
}

const ChannelConnecting: React.SFC<ChannelConnectingProps> = ({ msg }) => (
    <Message icon={true}>
        <Icon name='cog' loading={true} />
        <Message.Content>
            <Message.Header>Connecting to LiberTEM</Message.Header>
            {msg}
        </Message.Content>
    </Message>
)

export default ChannelConnecting;