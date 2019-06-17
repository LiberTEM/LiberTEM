import * as React from "react";
import { useState } from "react";
import { defaultDebounce } from "../../helpers";
import { AnalysisTypes, SumFrameParams } from "../../messages";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";

const useRoiPicker = ({ scanWidth, scanHeight, analysisId, run, enabled, jobIndex }: {
    scanWidth: number;
    scanHeight: number;
    enabled: boolean;
    jobIndex: number,
    analysisId: string;
    run: typeof analysisActions.Actions.run;
}) => {
    const minLength = Math.min(scanWidth, scanHeight);
    const [cx, setCx] = useState(scanWidth / 2);
    const [cy, setCy] = useState(scanHeight / 2);
    const [r, setR] = useState(minLength / 8);

    const roiParameters: SumFrameParams = {
        roi: {
            shape: "disk",
            cx,
            cy,
            r,
        },
    }

    React.useEffect(() => {
        if (enabled) {
            run(analysisId, jobIndex, {
                type: AnalysisTypes.SUM_FRAMES,
                parameters: roiParameters,
            });
        }
    }, [analysisId, enabled, jobIndex, cx, cy, r]);

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });

    const handleRChange = defaultDebounce((newR: number) => {
        setR(newR);
    });

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const sumRoiHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={scanWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(scanWidth, scanHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={scanWidth}
            onDragMove={cbToRadius(cx, cy, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={keepOnCY(cy)} />
    </>);

    const sumRoiWidgets = (
        <Disk cx={cx} cy={cy} r={r}
            imageWidth={scanWidth} imageHeight={scanHeight}
        />
    );

    return {
        roiParameters,
        sumRoiHandles,
        sumRoiWidgets,
    };
};

export { useRoiPicker };

