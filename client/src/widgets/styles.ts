import { CSSProperties } from "react";

export const defaultMaskStyles = (imageWidth: number): CSSProperties => ({
    fillOpacity: 0.3,
    fill: "red",
    strokeOpacity: 0.7,
    stroke: "red",
    strokeWidth: imageWidth / 128 / 3,
});