import * as React from "react";
import DraggableHandle from "./DraggableHandle";
import HandleParent from "./HandleParent";
import { defaultMaskStyles } from "./styles";

export interface DiskProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    r: number,
    image?: string,  // URL (can be from a blob via createObjectURL)
    onCenterChange?: (x: number, y: number) => void,
    onRChange?: (r: number) => void,
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

const Disk: React.SFC<DiskProps> = ({ imageWidth, imageHeight, cx, cy, r, image, onCenterChange, onRChange }) => {
    const rHandle = {
        x: cx - r,
        y: cy,
    }
    return (
        <svg width={imageWidth} height={imageHeight} viewBox={`0 0 ${imageWidth} ${imageHeight}`} style={{ border: "1px solid black" }}>
            {image ? <image xlinkHref={image} width={imageWidth} height={imageHeight} /> : null}
            <circle cx={cx} cy={cy} r={r} style={{ ...defaultMaskStyles }} />
            <HandleParent width={imageWidth} height={imageHeight}>
                <DraggableHandle x={cx} y={cy}
                    onDragMove={onCenterChange}
                    constraint={inRectConstraint(imageWidth, imageHeight)} />
                <DraggableHandle x={rHandle.x} y={rHandle.y}
                    onDragMove={cbToRadius(cx, cy, onRChange)}
                    constraint={keepOnCY(cy)} />
            </HandleParent>
        </svg>
    );
}

export default Disk;