import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Grid, Header, Segment } from "semantic-ui-react";
import { defaultDebounce } from "../../helpers";
import JobComponent from "../../job/Job";
import { Dataset, MaskDefRing } from "../../messages";
import Ring from "../../widgets/Ring";
import * as analysisActions from "../actions";
import { Analysis } from "../types";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    parameters: MaskDefRing,
    analysis: Analysis,
    dataset: Dataset,
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
    const { shape } = dataset;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    return (
        <>
            <Header attached="top">Ring analysis</Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            <Ring cx={parameters.cx} cy={parameters.cy} ri={parameters.ri} ro={parameters.ro}
                                imageWidth={imageWidth} imageHeight={imageHeight} onCenterChange={handleCenterChange} onRIChange={handleRIChange} onROChange={handleROChange} />
                        </Grid.Column>
                        <Grid.Column>
                            {currentJob !== "" ? <JobComponent job={currentJob} /> : null}
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>

    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(RingMaskAnalysis);