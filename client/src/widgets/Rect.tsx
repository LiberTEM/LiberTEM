import * as React from "react";
import { defaultMaskStyles } from "./styles";

export interface RectProps {
    imageWidth: number,
    imageHeight: number,
    x: number,
    y: number,
    width: number,
    height: number,
}

const Rect: React.SFC<RectProps> = ({ imageWidth, imageHeight, x, y, width, height }) => {
    return (
        <rect x={x + .5} y={y + .5} width={width} height={height} style={{ ...defaultMaskStyles(imageWidth) }} />
    );
}

export default Rect;