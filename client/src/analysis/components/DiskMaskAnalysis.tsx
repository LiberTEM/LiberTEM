import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { Grid, Header, Segment } from "semantic-ui-react";
import { defaultDebounce } from "../../helpers";
import JobComponent from "../../job/Job";
import { Dataset, MaskDefDisk } from "../../messages";
import Disk from "../../widgets/Disk";
import * as analysisActions from "../actions";
import { Analysis } from "../types";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    parameters: MaskDefDisk,
    analysis: Analysis,
    dataset: Dataset,
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

const DiskMaskAnalysis: React.SFC<MergedProps> = ({ analysis, dataset, parameters, handleRChange, handleCenterChange }) => {
    const { currentJob } = analysis;
    const { shape } = dataset;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    return (
        <>
            <Header as='h3' attached="top">Disk analysis</Header>
            <Segment attached={true}>
                <Grid columns={2}>
                    <Grid.Row>
                        <Grid.Column>
                            <Disk cx={parameters.cx} cy={parameters.cy} r={parameters.r}
                                imageWidth={imageWidth} imageHeight={imageHeight} onCenterChange={handleCenterChange} onRChange={handleRChange} />
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

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(DiskMaskAnalysis);