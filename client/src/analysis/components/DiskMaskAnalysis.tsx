import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";
import { getPreviewURL } from "../../dataset/api";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { DatasetState, MaskDefDisk } from "../../messages";
import Disk from "../../widgets/Disk";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    parameters: MaskDefDisk,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }));
        }),
        handleRChange: defaultDebounce((r: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { r }));
        }),
    }
}


type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const DiskMaskAnalysis: React.SFC<MergedProps> = ({ parameters, analysis, dataset, handleRChange, handleCenterChange }) => {
    const { currentJob } = analysis;
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const resultWidth = shape[1];
    const resultHeight = shape[0];

    return (
        <>
            <Header as='h3' attached="top">
                <Icon name="cog" />
                <Header.Content>Disk analysis</Header.Content>
            </Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            <Disk cx={parameters.cx} cy={parameters.cy} r={parameters.r}
                                image={getPreviewURL(dataset)}
                                imageWidth={imageWidth} imageHeight={imageHeight} onCenterChange={handleCenterChange} onRChange={handleRChange} />
                            <p>Disk: center=({parameters.cx},{parameters.cy}), r={parameters.r}</p>
                        </Grid.Column>
                        <Grid.Column>
                            <ResultList job={currentJob} width={resultWidth} height={resultHeight} />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(DiskMaskAnalysis);