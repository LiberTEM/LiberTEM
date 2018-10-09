import * as React from "react";
import { getPathArc } from "../helpers/svg";
import { cbToRadius, inRectConstraint, riConstraint, roConstraints } from "./constraints";
import DraggableHandle from "./DraggableHandle";
import HandleParent from "./HandleParent";
import { defaultMaskStyles } from "./styles";

export interface RingProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    ri: number,
    ro: number,
    image?: React.ReactElement<any>,
    onCenterChange?: (x: number, y: number) => void,
    onRIChange?: (r: number) => void,
    onROChange?: (r: number) => void,
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

    // see also: https://stackoverflow.com/a/37883328/540644
    const pathSpecs = [
        getPathArc({ x: cx, y: cy }, 90, 90, ro),
        getPathArc({ x: cx, y: cy }, 90, 90, ri)
    ]
    const pathSpec = pathSpecs.join(' ');
    return (
        <svg style={{ display: "block", border: "1px solid black", width: "100%", height: "auto" }} width={imageWidth} height={imageHeight} viewBox={`0 0 ${imageWidth} ${imageHeight}`}>
            {image}
            <path d={pathSpec} fillRule="evenodd" style={{ ...defaultMaskStyles(imageWidth) }} />
            <HandleParent width={imageWidth} height={imageHeight}>
                <DraggableHandle x={cx} y={cy}
                    imageWidth={imageWidth}
                    onDragMove={onCenterChange}
                    constraint={inRectConstraint(imageWidth, imageHeight)} />
                <DraggableHandle x={roHandle.x} y={roHandle.y}
                    imageWidth={imageWidth}
                    onDragMove={cbToRadius(cx, cy, onROChange)}
                    constraint={roConstraints(riHandle.x, cy)} />
                <DraggableHandle x={riHandle.x} y={riHandle.y}
                    imageWidth={imageWidth}
                    onDragMove={cbToRadius(cx, cy, onRIChange)}
                    constraint={riConstraint(roHandle.x, cy)} />
            </HandleParent>
        </svg>
    );
}

export default Ring;