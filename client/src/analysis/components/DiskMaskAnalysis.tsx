import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { AnalysisProps } from "../types";
import AnalysisLayoutTwoCol from "./AnalysisLayoutTwoCol";
import useDefaultFrameView from "./DefaultFrameView";
import Toolbar from "./Toolbar";


const DiskMaskAnalysis: React.SFC<AnalysisProps> = ({ analysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;

    const minLength = Math.min(imageWidth, imageHeight);
    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const [r, setR] = useState(minLength / 4);

    const handleCenterChange = (newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    };
    const handleRChange = setR;

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(cx, cy, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={keepOnCY(cy)} />
    </>);

    const frameViewWidgets = (
        <Disk cx={cx} cy={cy} r={r}
            imageWidth={imageWidth} imageHeight={imageHeight}
        />
    );

    const dispatch = useDispatch();

    const runAnalysis = () => {
        dispatch(analysisActions.Actions.run(analysis.id, 1, {
            type: AnalysisTypes.APPLY_DISK_MASK,
            parameters: {
                shape: "disk",
                cx, cy, r
            }
        }));
    };

    const {
        frameViewTitle, frameModeSelector,
        handles: resultHandles, widgets: resultWidgets
    } = useDefaultFrameView({
        scanWidth,
        scanHeight,
        analysisId: analysis.id,
    })

    const subtitle = <>{frameViewTitle} Disk: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), r={r.toFixed(2)}</>;

    const toolbar = <Toolbar analysis={analysis} onApply={runAnalysis} busyIdxs={[1]} />

    return (
        <AnalysisLayoutTwoCol
            title="Disk analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles} extraWidgets={frameViewWidgets}
                    jobIndex={0} analysis={analysis.id}
                    width={imageWidth} height={imageHeight}
                    selectors={frameModeSelector}
                />
            </>}
            right={<>
                <ResultList
                    jobIndex={1} analysis={analysis.id}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                    extraWidgets={resultWidgets}
                />
            </>}
            toolbar={toolbar}
        />
    );
}

export default DiskMaskAnalysis;