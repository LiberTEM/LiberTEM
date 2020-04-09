import 'semantic-ui-css/semantic.min.css';
import * as React from 'react';
import { Container, Icon, Modal, Popup } from 'semantic-ui-react';
import About from './About';
import ChannelStatus from './channel/components/ChannelStatus';
import { connect } from "react-redux";
import DatasetList from './dataset/components/DatasetList';
import ErrorList from './errors/components/ErrorList';
import logo from './images/LiberTEM logo-medium.png';
import { RootReducer } from "./store";

const mapStateToProps = (state: RootReducer) => {
    return {
        noOfDatasets: state.datasets.ids.length,
        isVisible: state.openDataset.formVisible,
    };
}

type MergedProps = ReturnType<typeof mapStateToProps>;

class App extends React.Component<MergedProps> {
    public render() {
        const { noOfDatasets, isVisible } = this.props;
        return (
            <Container style={{ margin: "5em 1em 5em 1em" }}>
                {(noOfDatasets || isVisible) ? 
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
                : ''}
                <ErrorList />
                <ChannelStatus>
                    <DatasetList />
                </ChannelStatus>
            </Container>
        );
    }

}

export default connect(mapStateToProps)(App);