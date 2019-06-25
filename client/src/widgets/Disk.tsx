import * as React from "react";
import { defaultMaskStyles } from "./styles";

export interface DiskProps {
    imageWidth: number,
    imageHeight: number,
    cx: number,
    cy: number,
    r: number,
}

const Disk: React.SFC<DiskProps> = ({ imageWidth, imageHeight, cx, cy, r }) => {
    return (
        <circle cx={cx + .5} cy={cy + .5} r={r} style={{ ...defaultMaskStyles(imageWidth) }} />
    );
}

export default Disk;