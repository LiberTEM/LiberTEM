import * as React from "react";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";

interface AnalysisLayoutThreeColProps {
    title: string,
    subtitle: React.ReactNode,
    left: React.ReactNode,
    mid: React.ReactNode,
    right: React.ReactNode,
    toolbar?: React.ReactNode,
    title1: string,
    title2: React.ReactNode,
    title3: string,
}

type MergedProps = AnalysisLayoutThreeColProps;

const AnalysisLayoutThreeCol: React.FC<MergedProps> = ({
    title, subtitle,
    left, mid, right,
    toolbar, title1, title2, title3
}) => (
    <>
        <Header as='h3' attached="top">
            <Icon name="cog" />
            <Header.Content>{title}</Header.Content>
        </Header>
        <Segment attached>
            <Grid columns={3}>
                <Grid.Row>
                    <Grid.Column width={5}>
                        <p>{title1}</p>
                    </Grid.Column>

                    <Grid.Column width={5}>
                        <p>{title2}</p>
                    </Grid.Column>

                    <Grid.Column width={6}>
                        <p>{title3}</p>
                    </Grid.Column>
                </Grid.Row>

                <Grid.Row>
                    <Grid.Column width={5}>
                        {left}
                    </Grid.Column>

                    <Grid.Column width={5}>
                        {mid}
                    </Grid.Column>

                    <Grid.Column width={6}>
                        {right}
                    </Grid.Column>

                </Grid.Row>

                <Grid.Row>
                    <Grid.Column width={10}>
                        <p>{subtitle}</p>
                    </Grid.Column>

                    <Grid.Column width={6} />
                </Grid.Row>
            </Grid>
        </Segment>
        {toolbar}
    </>
);

export default AnalysisLayoutThreeCol