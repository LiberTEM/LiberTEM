import * as React from "react";
import DraggableHandle from "./DraggableHandle";
import HandleParent from "./HandleParent";

export interface RingProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    ri: number,
    ro: number,
    image?: string,  // URL (can be from a blob via createObjectURL)
    onCenterChange?: (x: number, y: number) => void,
    onRIChange?: (r: number) => void,
    onROChange?: (r: number) => void,
}

const dist = (cx: number, cy: number, x: number, y: number) => {
    const dx = cx - x;
    const dy = cy - y;
    return Math.sqrt(dx * dx + dy * dy);
}

const cbToRadius = (cx: number, cy: number, cb: ((r: number) => void) | undefined) => (x: number, y: number) => cb && cb(dist(cx, cy, x, y))

const inRectConstraint = (width: number, height: number) => (p: Point2D) => {
    return {
        x: Math.max(0, Math.min(width, p.x)),
        y: Math.max(0, Math.min(height, p.y)),
    }
}

const keepOnCY = (cy: number) => (p: Point2D) => {
    return {
        x: p.x,
        y: cy,
    }
}

const keepXLargerThan = (otherX: number) => (p: Point2D) => {
    return {
        x: otherX > p.x ? otherX : p.x,
        y: p.y,
    }
}

const keepXSmallerThan = (otherX: number) => (p: Point2D) => {
    return {
        x: otherX < p.x ? otherX : p.x,
        y: p.y,
    }
}

const riConstraint = (outerPos: number, cy: number) => (p: Point2D) => {
    return keepXLargerThan(outerPos)(
        keepOnCY(cy)(p)
    );
}

const roConstraints = (innerPos: number, cy: number) => (p: Point2D) => {
    return keepXSmallerThan(innerPos)(
        keepOnCY(cy)(p)
    );
}

const Ring: React.SFC<RingProps> = ({ imageWidth, imageHeight, cx, cy, ri, ro, image, onCenterChange, onRIChange, onROChange }) => {
    const riHandle = {
        x: cx - ri,
        y: cy,
    }
    const roHandle = {
        x: cx - ro,
        y: cy,
    }
    return (
        <svg width={imageWidth} height={imageHeight} viewBox={`0 0 ${imageWidth} ${imageHeight}`} style={{ border: "1px solid black" }}>
            {image ? <image xlinkHref={image} width={imageWidth} height={imageHeight} /> : null}
            <circle cx={cx} cy={cy} r={ro} style={{ fill: "black" }} />
            <circle cx={cx} cy={cy} r={ri} style={{ fill: "white" }} />
            <HandleParent width={imageWidth} height={imageHeight}>
                <DraggableHandle x={cx} y={cy}
                    onDragMove={onCenterChange}
                    constraint={inRectConstraint(imageWidth, imageHeight)} />
                <DraggableHandle x={riHandle.x} y={riHandle.y}
                    onDragMove={cbToRadius(cx, cy, onRIChange)}
                    constraint={riConstraint(roHandle.x, cy)} />
                <DraggableHandle x={roHandle.x} y={roHandle.y}
                    onDragMove={cbToRadius(cx, cy, onROChange)}
                    constraint={roConstraints(riHandle.x, cy)} />
            </HandleParent>
        </svg>
    );
}

export default Ring;