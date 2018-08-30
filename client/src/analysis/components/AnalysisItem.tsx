import * as React from "react";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";
import ResultList from "../../job/components/ResultList";
import { DatasetState } from "../../messages";
import { AnalysisState } from "../types";
import PreviewModeSelector from "./PreviewModeSelector";
import Toolbar from "./Toolbar";

interface AnalysisItemProps {
    analysis: AnalysisState,
    dataset: DatasetState,
    title: string,
    subtitle: React.ReactNode,
}

const AnalysisItem: React.SFC<AnalysisItemProps> = ({ analysis, dataset, title, subtitle, children }) => {
    const { currentJob } = analysis;
    const { shape } = dataset.params;
    const resultWidth = shape[1];
    const resultHeight = shape[0];
    const pickCoords = analysis.preview.mode === "AVERAGE" ? null : `Pick: x=${analysis.preview.pick.x}, y=${analysis.preview.pick.y}`;

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
                            {children}
                            <PreviewModeSelector analysis={analysis} />
                            <p>{subtitle} {pickCoords}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <ResultList analysis={analysis.id} job={currentJob} width={resultWidth} height={resultHeight} />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    )
}

export default AnalysisItem;