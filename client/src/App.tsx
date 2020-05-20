import * as React from "react";
import "semantic-ui-css/semantic.min.css";
import { Container } from "semantic-ui-react";
import ChannelStatus from "./channel/components/ChannelStatus";
import DatasetList from "./dataset/components/DatasetList";
import ErrorList from "./errors/components/ErrorList";
import logo from "./images/LiberTEM logo-medium.png";
import HeaderMenu from "./Menu";

class App extends React.Component {
    public render() {
        return (
            <>
                <HeaderMenu />
                <Container style={{ margin: "5em 1em 5em 1em" }}>
                    <div style={{ display: "flex" }}>
                        <img src={logo} width="200" height="46" alt="LiberTEM" style={{ marginBottom: "20px" }} />
                    </div>
                    <ErrorList />
                    <ChannelStatus>
                        <DatasetList />
                    </ChannelStatus>
                </Container>
            </>
        );
    }
}

export default App;
