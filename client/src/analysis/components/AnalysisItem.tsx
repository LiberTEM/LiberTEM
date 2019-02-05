import * as React from "react";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";
import ResultList from "../../job/components/ResultList";
import { DatasetOpen } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import { AnalysisState } from "../types";
import FrameViewModeSelector from "./FrameViewModeSelector";
import PickHandle from "./PickHandle";
import Toolbar from "./Toolbar";

interface AnalysisItemProps {
    analysis: AnalysisState,
    dataset: DatasetOpen,
    title: string,
    subtitle: React.ReactNode,
    frameViewHandles?: HandleRenderFunction,
    frameViewWidgets?: React.ReactElement<SVGElement>,
}

type MergedProps = AnalysisItemProps;

const AnalysisItem: React.SFC<MergedProps> = ({
    frameViewHandles, frameViewWidgets,
    analysis, dataset, title, subtitle,
}) => {
    const { shape } = dataset.params;
    const resultWidth = shape[1];
    const resultHeight = shape[0];
    const frameWidth = shape[3];
    const frameHeight = shape[2];

    const resultHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (
        <PickHandle
            analysis={analysis}
            width={resultWidth}
            height={resultHeight}
            onDragStart={handleDragStart}
            onDrop={handleDrop}
        />
    );

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
                            <ResultList
                                extraHandles={frameViewHandles} extraWidgets={frameViewWidgets}
                                kind="FRAME"
                                analysis={analysis.id} width={frameWidth} height={frameHeight}
                                selectors={
                                    <FrameViewModeSelector analysis={analysis} />
                                } />
                            <p>{subtitle}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <ResultList
                                extraHandles={resultHandles}
                                kind='RESULT'
                                analysis={analysis.id} width={resultWidth} height={resultHeight}
                            />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    )
}

export default AnalysisItem