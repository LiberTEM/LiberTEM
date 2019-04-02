import * as React from "react";
import { connect } from "react-redux";
import * as analysisActions from '../../analysis/actions';
import { AnalysisTypes } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleDragStartFn, HandleDropFn } from "../../widgets/types";
import { AnalysisState } from "../types";

interface PickHandleProps {
    analysis: AnalysisState,
    width: number,
    height: number,
    onDragStart: HandleDragStartFn,
    onDrop: HandleDropFn,
}

const mapDispatchToProps = {
    updateParameters: analysisActions.Actions.updateParameters,
};

type MergedProps = DispatchProps<typeof mapDispatchToProps> & PickHandleProps;

const PickHandle: React.SFC<MergedProps> = ({
    analysis,
    width,
    height,
    updateParameters,
    onDragStart,
    onDrop,
}) => {
    if (analysis.frameDetails.type !== AnalysisTypes.PICK_FRAME) {
        return null;
    }
    const { x, y } = analysis.frameDetails.parameters;
    const onPickChange = (pickX: number, pickY: number) => {
        if (analysis.frameDetails.type !== AnalysisTypes.PICK_FRAME) {
            return;
        }
        const newX = Math.round(pickX);
        const newY = Math.round(pickY);
        if (x === newX && y === newY) {
            return;
        }
        updateParameters(analysis.id, {
            x: newX,
            y: newY,
        }, "FRAME");
    }


    return (
        <DraggableHandle x={x} y={y} withCross={true}
            imageWidth={width}
            onDragMove={onPickChange}
            parentOnDragStart={onDragStart}
            parentOnDrop={onDrop}
            constraint={inRectConstraint(width, height)} />
    );
}

export default connect(null, mapDispatchToProps)(PickHandle);