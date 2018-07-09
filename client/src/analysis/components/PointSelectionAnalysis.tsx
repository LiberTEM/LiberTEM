import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Grid, Header, Segment } from "semantic-ui-react";
import { getPreviewURL } from "../../dataset/api";
import { defaultDebounce } from "../../helpers";
import JobComponent from "../../job/components/Job";
import { DatasetState, PointDef } from "../../messages";
import Point from "../../widgets/Point";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    parameters: PointDef,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }));
        }),
    }
}


type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const PointSelectionAnalysis: React.SFC<MergedProps> = ({ parameters, analysis, dataset, handleCenterChange }) => {
    const { currentJob } = analysis;
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const resultWidth = shape[1];
    const resultHeight = shape[0];

    return (
        <>
            <Header as='h3' attached="top">Point analysis</Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            <Point cx={parameters.cx} cy={parameters.cy}
                                image={getPreviewURL(dataset)}
                                imageWidth={imageWidth} imageHeight={imageHeight} onCenterChange={handleCenterChange} />
                            <p>Point: center=({parameters.cx},{parameters.cy})</p>
                        </Grid.Column>
                        <Grid.Column>
                            <JobComponent job={currentJob} width={resultWidth} height={resultHeight} />
                        </Grid.Column>
                    </Grid.Row>
                </Grid>
            </Segment>
            <Toolbar analysis={analysis} />
        </>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(PointSelectionAnalysis);