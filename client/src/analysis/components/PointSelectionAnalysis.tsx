import * as React from "react";
import { connect, Dispatch } from "react-redux";
import { defaultDebounce } from "../../helpers";
import { DatasetState, PointDef } from "../../messages";
import Point from "../../widgets/Point";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";
import FrameView from "./FrameView";

interface AnalysisProps {
    parameters: PointDef,
    analysis: AnalysisState,
    dataset: DatasetState,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }, "RESULT"));
        }),
    }
}


type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const PointSelectionAnalysis: React.SFC<MergedProps> = ({ parameters, analysis, dataset, handleCenterChange }) => {
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const image = <FrameView dataset={dataset} analysis={analysis} />

    return (
        <AnalysisItem analysis={analysis} dataset={dataset} title="Point analysis" subtitle={
            <>Point: center=(x={parameters.cx.toFixed(2)}, y={parameters.cy.toFixed(2)})</>
        }>
            <Point cx={parameters.cx} cy={parameters.cy}
                image={image}
                imageWidth={imageWidth} imageHeight={imageHeight} onCenterChange={handleCenterChange} />
        </AnalysisItem>
    );
}

export default connect<{}, {}, AnalysisProps>(state => ({}), mapDispatchToProps)(PointSelectionAnalysis);