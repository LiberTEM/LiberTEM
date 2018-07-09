import * as React from "react";
import DraggableHandle from "./DraggableHandle";
import HandleParent from "./HandleParent";

export interface PointProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    image?: string,  // URL (can be from a blob via createObjectURL)
    onCenterChange?: (x: number, y: number) => void,
    onRChange?: (r: number) => void,
}

const inRectConstraint = (width: number, height: number) => (p: Point2D) => {
    return {
        x: Math.max(0, Math.min(width, p.x)),
        y: Math.max(0, Math.min(height, p.y)),
    }
}

const Point: React.SFC<PointProps> = ({ imageWidth, imageHeight, cx, cy, image, onCenterChange }) => {
    return (
        <svg style={{ border: "1px solid black", width: "100%", height: "auto" }} width={imageWidth} height={imageHeight} viewBox={`0 0 ${imageWidth} ${imageHeight}`}>
            {image ? <image xlinkHref={image} width={imageWidth} height={imageHeight} /> : null}
            <HandleParent width={imageWidth} height={imageHeight}>
                <DraggableHandle x={cx} y={cy} withCross={true}
                    onDragMove={onCenterChange}
                    constraint={inRectConstraint(imageWidth, imageHeight)} />
            </HandleParent>
        </svg>
    );
}

export default Point;