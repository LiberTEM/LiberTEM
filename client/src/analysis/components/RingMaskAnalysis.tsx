import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { defaultDebounce } from "../../helpers";
import { DatasetOpen, MaskDefRing } from "../../messages";
import Ring from "../../widgets/Ring";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";
import FrameView from "./FrameView";

interface AnalysisProps {
    parameters: MaskDefRing,
    analysis: AnalysisState,
    dataset: DatasetOpen,
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: AnalysisProps) => {
    return {
        handleCenterChange: defaultDebounce((cx: number, cy: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { cx, cy }, "RESULT"));
        }),
        handleRIChange: defaultDebounce((ri: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ri }, "RESULT"));
        }),
        handleROChange: defaultDebounce((ro: number) => {
            dispatch(analysisActions.Actions.updateParameters(ownProps.analysis.id, { ro }, "RESULT"));
        }),
    }
}

type MergedProps = AnalysisProps & ReturnType<typeof mapDispatchToProps>

const RingMaskAnalysis: React.SFC<MergedProps> = ({ analysis, dataset, parameters, handleCenterChange, handleRIChange, handleROChange }) => {
    const { shape } = dataset.params;
    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const image = <FrameView dataset={dataset} analysis={analysis} />

    return (
        <AnalysisItem analysis={analysis} dataset={dataset} title="Ring analysis" subtitle={
            <>Ring: center=(x={parameters.cx.toFixed(2)}, y={parameters.cy.toFixed(2)}), ri={parameters.ri.toFixed(2)}, ro={parameters.ro.toFixed(2)}</>
        }>
            <Ring cx={parameters.cx} cy={parameters.cy} ri={parameters.ri} ro={parameters.ro}
                imageWidth={imageWidth} imageHeight={imageHeight} image={image}
                onCenterChange={handleCenterChange} onRIChange={handleRIChange} onROChange={handleROChange} />
        </AnalysisItem>
    );
}

export default connect(null, mapDispatchToProps)(RingMaskAnalysis);