import * as React from "react";
import { connect } from "react-redux";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";
import ResultList from "../../job/components/ResultList";
import { JobReducerState } from "../../job/reducers";
import { AnalysisTypes, DatasetState } from "../../messages";
import { RootReducer } from "../../store";
import BusyWrapper from "../../widgets/BusyWrapper";
import { AnalysisState } from "../types";
import FrameViewModeSelector from "./FrameViewModeSelector";
import Toolbar from "./Toolbar";

interface AnalysisItemProps {
    analysis: AnalysisState,
    dataset: DatasetState,
    title: string,
    subtitle: React.ReactNode,
}

type MergedProps = AnalysisItemProps & ReturnType<typeof mapStateToProps>;

const AnalysisItem: React.SFC<MergedProps> = ({ frameJob, analysis, dataset, title, subtitle, children }) => {
    const { shape } = dataset.params;
    const resultWidth = shape[1];
    const resultHeight = shape[0];
    const pickCoords = analysis.frameDetails.type === AnalysisTypes.SUM_FRAMES ? null : `Pick: x=${analysis.frameDetails.parameters.x}, y=${analysis.frameDetails.parameters.y}`;

    const frameViewBusy = frameJob !== undefined ? frameJob.running !== "DONE" : false;

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
                            <BusyWrapper busy={frameViewBusy}>
                                {children}
                            </BusyWrapper>
                            <FrameViewModeSelector analysis={analysis} />
                            <p>{subtitle} {pickCoords}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <ResultList analysis={analysis.id} width={resultWidth} height={resultHeight} />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    )
}

const getFrameJob = (analysis: AnalysisState, jobs: JobReducerState) => {
    const frameJobId = analysis.jobs.FRAME;
    if (frameJobId === undefined) {
        return;
    }
    return jobs.byId[frameJobId];
}

const mapStateToProps = (state: RootReducer, ownProps: AnalysisItemProps) => {
    return {
        frameJob: getFrameJob(ownProps.analysis, state.jobs)
    }
}

export default connect(mapStateToProps)(AnalysisItem);