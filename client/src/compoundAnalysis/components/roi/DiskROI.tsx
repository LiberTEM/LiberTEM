import { useState } from "react";

import { FrameParams } from "../../../messages";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../../widgets/constraints";
import Disk from "../../../widgets/Disk";
import DraggableHandle from "../../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../../widgets/types";

const useDiskROI = ({ scanWidth, scanHeight }: {
    scanWidth: number;
    scanHeight: number;
}) => {
    const minLength = Math.min(scanWidth, scanHeight);
    const [cx, setCx] = useState(scanWidth / 2);
    const [cy, setCy] = useState(scanHeight / 2);
    const [r, setR] = useState(minLength / 8);


    const diskRoiParameters: FrameParams = {
        roi: {
            shape: "disk",
            cx,
            cy,
            r,
        },
    }


    const handleCenterChange = (newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    };

    const handleRChange = (newR: number) => {
        setR(newR);
    };

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const diskRoiHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
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

    const diskRoiWidgets = (
        <Disk cx={cx} cy={cy} r={r}
            imageWidth={scanWidth}
        />
    );

    return {
        diskRoiParameters,
        diskRoiHandles,
        diskRoiWidgets,
    };
};

export { useDiskROI };

