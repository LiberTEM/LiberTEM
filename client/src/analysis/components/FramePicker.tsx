import React from "react";
import { useDispatch } from "react-redux";
import { AnalysisParameters, AnalysisTypes } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";

const useFramePicker = ({
    enabled, scanWidth, scanHeight, jobIndex, analysisId, cx, cy, setCx, setCy
}: {
    enabled: boolean, scanWidth: number, scanHeight: number,
    jobIndex: number, analysisId: string,
    cx: number, cy: number, setCx: (newCx: number) => void, setCy: (newCy: number) => void,
}) => {

    const dispatch = useDispatch();

    React.useEffect(() => {
        if (enabled) {
            const params: AnalysisParameters = {
                x: cx,
                y: cy,
            };

            dispatch(analysisActions.Actions.run(analysisId, jobIndex, {
                type: AnalysisTypes.PICK_FRAME,
                parameters: params,
            }))
        }
    }, [analysisId, cx, cy, enabled, jobIndex, dispatch]);

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
        <DraggableHandle x={cx} y={cy} withCross={true}
            imageWidth={scanWidth}
            onDragMove={onPickChange}
            parentOnDragStart={onDragStart}
            parentOnDrop={onDrop}
            constraint={inRectConstraint(scanWidth, scanHeight)} />
    )

    return { coords: { cx, cy }, handles: renderPickHandle };
}

export default useFramePicker;