import * as React from 'react';
import 'semantic-ui-css/semantic.min.css';
import { Container, Header } from 'semantic-ui-react';
import ChannelStatus from './channel/components/ChannelStatus';
import DatasetList from './dataset/components/DatasetList';

class App extends React.Component {
    public render() {
        return (
            <Container style={{ marginTop: "5em", marginBottom: "5em", }}>
                <Header as='h1'>LiberTEM</Header>
                <ChannelStatus>
                    <DatasetList />
                </ChannelStatus>
            </Container>
        );
    }
}

export default App;
