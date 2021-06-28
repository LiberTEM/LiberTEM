import React from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";

const useFFTFramePicker = ({
    enabled, scanWidth, scanHeight, analysisIndex, compoundAnalysisId,
    cx, cy, real_rad, real_centerx, real_centery, setCx, setCy
}: {
    enabled: boolean, scanWidth: number, scanHeight: number,
    analysisIndex: number, compoundAnalysisId: string,
    cx: number, cy: number, setCx: (newCx: number) => void, setCy: (newCy: number) => void,
    real_rad: number | null, real_centerx: number | null, real_centery: number | null
}) => {

    const dispatch = useDispatch();

    React.useEffect(() => {
        if (enabled) {
            dispatch(compoundAnalysisActions.Actions.run(compoundAnalysisId, analysisIndex, {
                analysisType: AnalysisTypes.PICK_FFT_FRAME,
                parameters: { x: cx, y: cy, real_rad, real_centerx, real_centery },
            }));
        }
    }, [compoundAnalysisId, cx, cy, enabled, analysisIndex, real_rad, real_centerx, real_centery, dispatch]);

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

export default useFFTFramePicker;