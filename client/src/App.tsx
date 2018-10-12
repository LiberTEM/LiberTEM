import * as React from 'react';
import 'semantic-ui-css/semantic.min.css';
import { Container, Header, Icon, Modal, Popup } from 'semantic-ui-react';
import About from './About';
import ChannelStatus from './channel/components/ChannelStatus';
import DatasetList from './dataset/components/DatasetList';
import ErrorList from './errors/components/ErrorList';

class App extends React.Component {
    public render() {
        return (
            <Container style={{ margin: "5em 1em 5em 1em" }}>
                <Header as='h1'>
                    LiberTEM
                    {' '}
                    <Modal trigger={
                        <Header.Content>
                            <Icon name="info circle" size="small" link={true} />
                        </Header.Content>
                    }>
                        <Popup.Header>About LiberTEM</Popup.Header>
                        <Popup.Content>
                            <About />
                        </Popup.Content>
                    </Modal>
                </Header>
                <ErrorList />
                <ChannelStatus>
                    <DatasetList />
                </ChannelStatus>
            </Container>
        );
    }
}

export default App;
