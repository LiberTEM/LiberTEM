import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { defaultDebounce } from "../../helpers";
import { DatasetState, MaskDefDisk } from "../../messages";
import Disk from "../../widgets/Disk";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";
import FrameView from "./FrameView";

interface AnalysisProps {
    parameters: MaskDefDisk,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }, "RESULT"));
        }),
        handleRChange: defaultDebounce((r: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { r }, "RESULT"));
        }),
    }
}


type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const DiskMaskAnalysis: React.SFC<MergedProps> = ({ parameters, analysis, dataset, handleRChange, handleCenterChange }) => {
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const image = <FrameView dataset={dataset} analysis={analysis} />

    return (
        <AnalysisItem analysis={analysis} dataset={dataset} title="Disk analysis" subtitle={
            <>Disk: center=(x={parameters.cx.toFixed(2)}, y={parameters.cy.toFixed(2)}), r={parameters.r.toFixed(2)}</>
        }>
            <Disk cx={parameters.cx} cy={parameters.cy} r={parameters.r}
                image={image}
                imageWidth={imageWidth} imageHeight={imageHeight} onCenterChange={handleCenterChange} onRChange={handleRChange} />
        </AnalysisItem>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(DiskMaskAnalysis);