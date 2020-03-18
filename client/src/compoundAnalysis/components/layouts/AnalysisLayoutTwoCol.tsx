import * as React from "react";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";

interface AnalysisLayoutTwoColProps {
    title: string,
    subtitle: React.ReactNode,
    left: React.ReactNode,
    right: React.ReactNode,
    toolbar?: React.ReactNode,
}

type MergedProps = AnalysisLayoutTwoColProps;

const AnalysisLayoutTwoCol: React.SFC<MergedProps> = ({
    title, subtitle,
    left, right,
    toolbar,
}) => {
    return (
        <>
            <Header as='h3' attached="top">
                <Icon name="cog" />
                <Header.Content>{title}</Header.Content>
            </Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            {left}
                            <p>{subtitle}</p>
                        </Grid.Column>
                        <Grid.Column>
                            {right}
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            {toolbar}
        </>
    )
}

export default AnalysisLayoutTwoCol