import React from "react";
import { AnalysisTypes } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";

const useFramePicker = ({
    enabled, scanWidth, scanHeight, jobIndex, analysisId, run
}: {
    enabled: boolean, scanWidth: number, scanHeight: number,
    jobIndex: number, analysisId: string, run: typeof analysisActions.Actions.run
}) => {
    const [cx, setCx] = React.useState(Math.round(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.round(scanHeight / 2));

    React.useEffect(() => {
        if (enabled) {
            run(analysisId, jobIndex, {
                type: AnalysisTypes.PICK_FRAME,
                parameters: {
                    x: cx,
                    y: cy,
                },
            });
        }
    }, [analysisId, cx, cy, enabled, jobIndex]);

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