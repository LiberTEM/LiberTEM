import * as React from "react";
import { defaultMaskStyles } from "./styles";

export interface DiskProps {
    imageWidth: number,
    cx: number,
    cy: number,
    r: number,
}

const Disk: React.FC<DiskProps> = ({ imageWidth, cx, cy, r }) => (
    <circle cx={cx + .5} cy={cy + .5} r={r} style={{ ...defaultMaskStyles(imageWidth) }} />
);

export default Disk;
