import * as React from "react";
import { getPathArc } from "../helpers/svg";
import { defaultMaskStyles } from "./styles";

export interface RingProps {
    imageWidth: number,
    cx: number,
    cy: number,
    ri: number,
    ro: number,
}

const Ring: React.FC<RingProps> = ({ imageWidth, cx, cy, ri, ro }) => {
    // see also: https://stackoverflow.com/a/37883328/540644
    const pathSpecs = [
        getPathArc({ x: cx + .5, y: cy + .5 }, 90, 90, ro),
        getPathArc({ x: cx + .5, y: cy + .5 }, 90, 90, ri)
    ]
    const pathSpec = pathSpecs.join(' ');

    return (
        <path d={pathSpec} fillRule="evenodd" style={{ ...defaultMaskStyles(imageWidth) }} />
    );
}

export default Ring;
