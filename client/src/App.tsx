import * as React from 'react';
import 'semantic-ui-css/semantic.min.css';
import { Button, Container, Grid, Header, Icon, Label } from 'semantic-ui-react';
import JobList from './job/JobList';


class App extends React.Component {
    public render() {
        return (
            <Container style={{ marginTop: "5em" }}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column stretched={true}>
                            <Header as='h1'>LiberTEM</Header>
                        </Grid.Column>
                    </Grid.Row>
                    <Grid.Row>
                        <Grid.Column stretched={true}>
                            <Button as='div' labelPosition='right'>
                                <Button icon={true}>
                                    <Icon name='add' />
                                </Button>
                                <Label as='a' basic={true}>
                                    Load Dataset
                                </Label>
                            </Button>
                        </Grid.Column>
                    </Grid.Row>
                    <Grid.Row>
                        <Grid.Column stretched={true}>
                            <JobList/>
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Container>
        );
    }
}

export default App;
