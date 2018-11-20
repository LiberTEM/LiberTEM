import * as React from "react";
import { connect } from "react-redux";
import { Header, List } from "semantic-ui-react";
import { RootReducer } from "./store";

const mapStateToProps = (state: RootReducer) => {
    return {
        version: state.config.version,
        revision: state.config.revision,
    }
}

type MergedProps = ReturnType<typeof mapStateToProps>;

const About: React.SFC<MergedProps> = ({ version, revision }) => {
    return (
        <>
            <Header as="h3">This is LiberTEM version {version} (revision {revision.slice(0, 8)})</Header>
            <List>
                <List.Item>
                    <List.Icon name="github" />
                    <List.Content>
                        Find us on <a href="https://github.com/LiberTEM/LiberTEM">GitHub</a>
                    </List.Content>
                </List.Item>
                <List.Item>
                    <List.Icon name="bug" />
                    <List.Content>
                        Found a bug? Got a feature request? Please <a href="https://github.com/LiberTEM/LiberTEM/issues/new">open an issue!</a>
                    </List.Content>
                </List.Item>
                <List.Item>
                    <List.Icon name="legal" />
                    <List.Content>
                        LiberTEM is licensed under the <a href="https://github.com/LiberTEM/LiberTEM/blob/master/LICENSE">GNU General Public License v3.0</a>
                    </List.Content>
                </List.Item>
                <List.Item>
                    <List.Icon name="book" />
                    <List.Content>
                        Read <a href="https://libertem.github.io/LiberTEM/">the documentation</a>
                    </List.Content>
                </List.Item>
                <List.Item>
                    <List.Icon name="user" />
                    <List.Content>
                        Read <a href="https://libertem.github.io/LiberTEM/acknowledgments.html">the acknowledgments</a>
                    </List.Content>
                </List.Item>
                <List.Item>
                    <List.Icon name="gitter" />
                    <List.Content>
                        Join our <a href="https://gitter.im/LiberTEM/Lobby">chat on gitter!</a>
                    </List.Content>
                </List.Item>
                <List.Item>
                    <List.Icon name="linkify" />
                    <List.Content>
                    <a href="https://doi.org/10.5281/zenodo.1478763"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1478763.svg" alt="DOI"/></a>
                    </List.Content>
                </List.Item>
            </List>
        </>
    );
}

export default connect(mapStateToProps)(About);