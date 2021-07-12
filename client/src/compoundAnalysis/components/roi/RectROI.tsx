import { useState } from "react";
import { RectRoiParams } from "../../../messages";
import DraggableHandle from "../../../widgets/DraggableHandle";
import Rect from "../../../widgets/Rect";
import { HandleRenderFunction } from "../../../widgets/types";



const useRectROI = ({ scanWidth, scanHeight }: {
    scanWidth: number;
    scanHeight: number;
}) => {
    const minLength = Math.min(scanWidth, scanHeight);
    const [x, setx] = useState(scanWidth / 2);
    const [y, sety] = useState(scanHeight / 2);
    const [width, setwidth] = useState(minLength / 8);
    const [height, setheight] = useState(minLength / 8);


    const rectRoiParameters: {roi: RectRoiParams} = {
        roi: {
            shape: "rect",
            x,
            y,
            width,
            height,
        },
    }


    const handleCornerChange = (newx: number, newy: number) => {
        setx(newx);
        sety(newy);
    };

    const handleShapeChange = (newx: number, newy: number) => {
        setwidth(newx - x);
        setheight(newy - y);
    };

    const smthHandle = {
        x: x + width,
        y: y + height,
    }

    const rectRoiHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={x} y={y}
            imageWidth={scanWidth}
            onDragMove={handleCornerChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop} />
        <DraggableHandle x={smthHandle.x} y={smthHandle.y}
            imageWidth={scanWidth}
            onDragMove={handleShapeChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop} />
    </>);

    const rectRoiWidgets = (
        <Rect x={x} y={y} width={width} height={height}
            imageWidth={scanWidth} imageHeight={scanHeight}
        />
    );

    return {
        rectRoiParameters,
        rectRoiHandles,
        rectRoiWidgets,
    };
};

export { useRectROI };

