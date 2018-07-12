import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Grid, Header, Icon, Segment } from "semantic-ui-react";
import { getPreviewURL } from "../../dataset/api";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { DatasetState, MaskDefRing } from "../../messages";
import Ring from "../../widgets/Ring";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    parameters: MaskDefRing,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            // FIXME: updateParameters doesn't seem to be typed strong enough
            // the following doesn't raise a type error:
            // dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { foo: "bar" }));
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }));
        }),
        handleRIChange: defaultDebounce((ri: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ri }));
        }),
        handleROChange: defaultDebounce((ro: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ro }));
        }),
    }
}

type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const RingMaskAnalysis: React.SFC<MergedProps> = ({ analysis, dataset, parameters, handleCenterChange, handleRIChange, handleROChange }) => {
    const { currentJob } = analysis;
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];
    const previewURL = getPreviewURL(dataset);

    const resultWidth = shape[1];
    const resultHeight = shape[0];

    return (
        <>
            <Header as='h3' attached="top">
                <Icon name="cog" />
                <Header.Content>Ring analysis</Header.Content>
            </Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            <Ring cx={parameters.cx} cy={parameters.cy} ri={parameters.ri} ro={parameters.ro}
                                imageWidth={imageWidth} imageHeight={imageHeight} image={previewURL}
                                onCenterChange={handleCenterChange} onRIChange={handleRIChange} onROChange={handleROChange}
                            />
                            <p>Ring: center=({parameters.cx},{parameters.cy}), ri={parameters.ri}, ro={parameters.ro}</p>
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

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(RingMaskAnalysis);