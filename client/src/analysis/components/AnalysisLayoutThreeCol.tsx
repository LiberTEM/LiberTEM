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
const title2 = "Masking out of zero-order diffraction peak in real space";
const title1 ="Masking of intergation region in Fourier space";
const title3 ="Result of analysis"
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
                            <p>{subtitle}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <p>{title2}</p>
                            {mid}
                            <p>{subtitle}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <p>{title3}</p>
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