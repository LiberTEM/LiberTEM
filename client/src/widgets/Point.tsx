import * as React from "react";
import { inRectConstraint } from "./constraints";
import DraggableHandle from "./DraggableHandle";
import HandleParent from "./HandleParent";

export interface PointProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    image?: React.ReactElement<any>,
    onCenterChange?: (x: number, y: number) => void,
    onRChange?: (r: number) => void,
}

const Point: React.SFC<PointProps> = ({ imageWidth, imageHeight, cx, cy, image, onCenterChange }) => {
    return (
        <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={imageWidth} height={imageHeight} viewBox={`0 0 ${imageWidth} ${imageHeight}`}>
            {image}
            <HandleParent width={imageWidth} height={imageHeight}>
                <DraggableHandle x={cx} y={cy} withCross={true}
                    onDragMove={onCenterChange}
                    imageWidth={imageWidth}
                    constraint={inRectConstraint(imageWidth, imageHeight)} />
            </HandleParent>
        </svg>
    );
}

export default Point;