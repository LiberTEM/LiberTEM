import React from "react";
import { useDispatch } from "react-redux";
import { AnalysisParameters, AnalysisTypes } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";

const useFramePicker = ({
    enabled, scanWidth, scanHeight, analysisIndex, compoundAnalysisId, cx, cy, setCx, setCy
}: {
    enabled: boolean, scanWidth: number, scanHeight: number,
    analysisIndex: number, compoundAnalysisId: string,
    cx: number, cy: number, setCx: (newCx: number) => void, setCy: (newCy: number) => void,
}) => {

    const dispatch = useDispatch();

    React.useEffect(() => {
        if (enabled) {
            const params: AnalysisParameters = {
                x: cx,
                y: cy,
            };

            dispatch(compoundAnalysisActions.Actions.run(compoundAnalysisId, analysisIndex, {
                analysisType: AnalysisTypes.PICK_FRAME,
                parameters: params,
            }))
        }
    }, [compoundAnalysisId, cx, cy, enabled, analysisIndex, dispatch]);

    const onPickChange = (pickX: number, pickY: number) => {
        const newX = Math.round(pickX);
        const newY = Math.round(pickY);
        if (cx === newX && cy === newY) {
            return;
        }
        setCx(newX);
        setCy(newY);
    }

    const renderPickHandle: HandleRenderFunction = (onDragStart, onDrop) => (
        <DraggableHandle x={cx} y={cy} withCross
            imageWidth={scanWidth}
            onDragMove={onPickChange}
            parentOnDragStart={onDragStart}
            parentOnDrop={onDrop}
            constraint={inRectConstraint(scanWidth, scanHeight)} />
    )

    return { coords: { cx, cy }, handles: renderPickHandle };
}

export default useFramePicker;