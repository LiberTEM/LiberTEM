import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { defaultDebounce } from "../../helpers";
import { DatasetOpen, MaskDefRing } from "../../messages";
import { cbToRadius, inRectConstraint, riConstraint, roConstraints } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import Ring from "../../widgets/Ring";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";

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

    const { cx, cy, ri, ro } = parameters;

    const riHandle = {
        x: cx - ri,
        y: cy,
    }
    const roHandle = {
        x: cx - ro,
        y: cy,
    }

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDrop={handleDrop}
            parentOnDragStart={handleDragStart}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={roHandle.x} y={roHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(cx, cy, handleROChange)}
            parentOnDrop={handleDrop}
            parentOnDragStart={handleDragStart}
            constraint={roConstraints(riHandle.x, cy)} />
        <DraggableHandle x={riHandle.x} y={riHandle.y}
            imageWidth={imageWidth}
            parentOnDrop={handleDrop}
            parentOnDragStart={handleDragStart}
            onDragMove={cbToRadius(cx, cy, handleRIChange)}
            constraint={riConstraint(roHandle.x, cy)} />
    </>);

    const frameViewWidgets = (
        <Ring cx={parameters.cx} cy={parameters.cy} ri={parameters.ri} ro={parameters.ro}
            imageWidth={imageWidth} />
    )

    const subtitle = (
        <>Ring: center=(x={parameters.cx.toFixed(2)}, y={parameters.cy.toFixed(2)}), ri={parameters.ri.toFixed(2)}, ro={parameters.ro.toFixed(2)}</>
    )


    return (
        <AnalysisItem analysis={analysis} dataset={dataset}
            title="Ring analysis" subtitle={subtitle}
            frameViewHandles={frameViewHandles} frameViewWidgets={frameViewWidgets}
        />
    );
}

export default connect(null, mapDispatchToProps)(RingMaskAnalysis);