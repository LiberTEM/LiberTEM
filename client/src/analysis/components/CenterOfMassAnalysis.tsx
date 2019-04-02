import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { defaultDebounce } from "../../helpers";
import { CenterOfMassParams, DatasetOpen } from "../../messages";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import { DraggableHandle } from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";

interface AnalysisProps {
    parameters: CenterOfMassParams,
    analysis: AnalysisState,
    dataset: DatasetOpen,
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

const CenterOfMassAnalysis: React.SFC<MergedProps> = ({ parameters, analysis, dataset, handleRChange, handleCenterChange }) => {
    const { shape } = dataset.params;

    const imageWidth = shape[3];
    const imageHeight = shape[2];

    const { cx, cy, r } = parameters;

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(cx, cy, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={keepOnCY(cy)} />
    </>);

    const frameViewWidgets = (
        <Disk cx={parameters.cx} cy={parameters.cy} r={parameters.r}
            imageWidth={imageWidth} imageHeight={imageHeight} />
    )

    const subtitle = <>Disk: center=(x={parameters.cx.toFixed(2)}, y={parameters.cy.toFixed(2)}), r={parameters.r.toFixed(2)}</>;

    return (
        <AnalysisItem analysis={analysis} dataset={dataset}
            title="COM analysis" subtitle={subtitle}
            frameViewHandles={frameViewHandles} frameViewWidgets={frameViewWidgets}
        />
    );
}

export default connect(null, mapDispatchToProps)(CenterOfMassAnalysis);