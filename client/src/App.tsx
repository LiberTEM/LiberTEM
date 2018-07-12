import * as React from 'react';
import 'semantic-ui-css/semantic.min.css';
import { Header } from 'semantic-ui-react';
import ChannelStatus from './channel/components/ChannelStatus';
import DatasetList from './dataset/components/DatasetList';
import ErrorList from './errors/components/ErrorList';

class App extends React.Component {
    public render() {
        return (
            <div style={{ margin: "5em 1em 5em 1em" }}>
                <Header as='h1'>LiberTEM</Header>
                <ErrorList />
                <ChannelStatus>
                    <DatasetList />
                </ChannelStatus>
            </div>
        );
    }
}

export default App;
