import * as React from "react";
import { connect } from "react-redux";
import { Dispatch } from "redux";
import { defaultDebounce } from "../../helpers";
import { DatasetOpen, PointDef } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisItem from "./AnalysisItem";

interface AnalysisProps {
    parameters: PointDef,
    analysis: AnalysisState,
    dataset: DatasetOpen,
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

    const { cx, cy } = parameters;

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy} withCross={true}
            onDragMove={handleCenterChange}
            imageWidth={imageWidth}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
    </>);

    const subtitle = (
        <>Point: center=(x={parameters.cx.toFixed(2)}, y={parameters.cy.toFixed(2)})</>
    )

    return (
        <AnalysisItem analysis={analysis} dataset={dataset}
            title="Point analysis" subtitle={subtitle}
            frameViewHandles={frameViewHandles}
        />
    );
}

export default connect(null, mapDispatchToProps)(PointSelectionAnalysis);