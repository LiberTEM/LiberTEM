import * as React from 'react';
import 'semantic-ui-css/semantic.min.css';
import { Container, Icon, Modal, Popup } from 'semantic-ui-react';
import About from './About';
import ChannelStatus from './channel/components/ChannelStatus';
import DatasetList from './dataset/components/DatasetList';
import ErrorList from './errors/components/ErrorList';
import logo from './images/LiberTEM logo-medium.png';

class App extends React.Component {
    public render() {
        return (
            <Container style={{ margin: "5em 1em 5em 1em" }}>
                <div style={{ display: "flex" }}>
                    <img src={logo} width="200" height="46" alt="LiberTEM" style={{ marginBottom: "20px" }} />
                    {' '}
                    <Modal trigger={
                        <Icon name="info circle" link={true} style={{ alignSelf: "flex-start" }} />
                    }>
                        <Popup.Header>About LiberTEM</Popup.Header>
                        <Popup.Content>
                            <About />
                        </Popup.Content>
                    </Modal>
                </div>
                <ErrorList />
                <ChannelStatus>
                    <DatasetList />
                </ChannelStatus>
            </Container>
        );
    }
}

export default App;
