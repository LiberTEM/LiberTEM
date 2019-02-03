import * as React from "react";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";
import ResultList from "../../job/components/ResultList";
import { DatasetOpen } from "../../messages";
import { AnalysisState } from "../types";
import FrameViewModeSelector from "./FrameViewModeSelector";
import Toolbar from "./Toolbar";

interface AnalysisItemProps {
    analysis: AnalysisState,
    dataset: DatasetOpen,
    title: string,
    subtitle: React.ReactNode,
    frameViewHandles?: React.ReactElement<SVGElement>,
    frameViewWidgets?: React.ReactElement<SVGElement>,
}

type MergedProps = AnalysisItemProps;

const AnalysisItem: React.SFC<MergedProps> = ({ frameViewHandles, analysis, dataset, title, subtitle, children }) => {
    const { shape } = dataset.params;
    const resultWidth = shape[1];
    const resultHeight = shape[0];
    const frameWidth = shape[3];
    const frameHeight = shape[2];

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
                            <FrameViewModeSelector analysis={analysis} />
                            <ResultList extraHandles={frameViewHandles} kind='FRAME' analysis={analysis.id} width={frameWidth} height={frameHeight} >
                                {children}
                            </ResultList>
                            <p>{subtitle}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <ResultList kind='RESULT' analysis={analysis.id} width={resultWidth} height={resultHeight} />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    )
}

export default AnalysisItem