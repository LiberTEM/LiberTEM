import * as React from "react";
import { cbToRadius, inRectConstraint, keepOnCY } from "./constraints";
import DraggableHandle from "./DraggableHandle";
import HandleParent from "./HandleParent";
import { handleKeyEvent, ModifyCoords } from "./kbdHandler";
import { defaultMaskStyles } from "./styles";

export interface DiskProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    r: number,
    image?: React.ReactElement<any>,
    onCenterChange?: (x: number, y: number) => void,
    onRChange?: (r: number) => void,
}

const Disk: React.SFC<DiskProps> = ({ imageWidth, imageHeight, cx, cy, r, image, onCenterChange, onRChange }) => {
    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const myKeyEvent = (e: React.KeyboardEvent<SVGElement>) => {
        const update = (fn: ModifyCoords) => {
            if (onCenterChange) {
                const newCoords = fn(cx, cy);
                onCenterChange(newCoords.x, newCoords.y);
            };
        }
        handleKeyEvent(e, update)
    }

    return (
        <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={imageWidth} height={imageHeight} viewBox={`0 0 ${imageWidth} ${imageHeight}`}>
            {image}
            <circle cx={cx} cy={cy} r={r} style={{ ...defaultMaskStyles(imageWidth) }} />
            <HandleParent width={imageWidth} height={imageHeight} onKeyboardEvent={myKeyEvent}>
                <DraggableHandle x={cx} y={cy}
                    imageWidth={imageWidth}
                    onDragMove={onCenterChange}
                    constraint={inRectConstraint(imageWidth, imageHeight)} />
                <DraggableHandle x={rHandle.x} y={rHandle.y}
                    imageWidth={imageWidth}
                    onDragMove={cbToRadius(cx, cy, onRChange)}
                    constraint={keepOnCY(cy)} />
            </HandleParent>
        </svg>
    );
}

export default Disk;