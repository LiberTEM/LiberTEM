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

const Rect: React.FC<RectProps> = ({ imageWidth, x, y, width, height }) => {
    let ymin: number;
    let xmin: number;
    if (height*width > 0) {
    ymin = Math.min(y, y+height);
    xmin = Math.min(x, x+width);}
    else if (height > 0 && width < 0) 
    {ymin = y;
    xmin = x+width;}
    else 
    {ymin = y+height;
    xmin = x;}
    return (
        <rect x={xmin + .5} y={ymin + .5} width={Math.abs(width)} height={Math.abs(height)} style={{ ...defaultMaskStyles(imageWidth) }} />
    );
}

export default Rect;

