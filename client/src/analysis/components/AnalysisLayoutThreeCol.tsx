import * as React from "react";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";

interface AnalysisLayoutTwoColProps {
    title: string,
    subtitle: React.ReactNode,
    left: React.ReactNode,
    mid: React.ReactNode,
    right: React.ReactNode,
    toolbar?: React.ReactNode,
}

type MergedProps = AnalysisLayoutTwoColProps;
const title1="yhfgkjdsh"
const AnalysisLayoutTwoCol: React.SFC<MergedProps> = ({
    title, subtitle,
    left, mid, right,
    toolbar,
}) => {

    return (
        <>
            <Header as='h3' attached="top">
                <Icon name="cog" />
                <Header.Content>{title}</Header.Content>
            </Header>
            <Segment attached={true}>
                <Grid columns={3}>
                    <Grid.Row>
                        <Grid.Column>
                            <p>{title1}</p>
                            {left}
                            <p>{subtitle}kjhljhio</p>
                        </Grid.Column>
                        <Grid.Column>
                            <p>{title1}</p>
                            {mid}
                            <p>{subtitle}lkjioljoi</p>
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