import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { cbToRadius, inRectConstraint, riConstraint, roConstraints } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import Ring from "../../widgets/Ring";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { CompoundAnalysisProps } from "../types";
import useDefaultFrameView from "./DefaultFrameView";
import AnalysisLayoutTwoCol from "./layouts/AnalysisLayoutTwoCol";
import Toolbar from "./Toolbar";

const RingMaskAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);

    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const [ri, setRi] = useState(minLength / 4);
    const [ro, setRo] = useState(minLength / 2);

    const riHandle = {
        x: cx - ri,
        y: cy,
    }
    const roHandle = {
        x: cx - ro,
        y: cy,
    }

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });
    const handleRIChange = defaultDebounce(setRi);
    const handleROChange = defaultDebounce(setRo);

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDrop={handleDrop}
            parentOnDragStart={handleDragStart}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={roHandle.x} y={roHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(cx, cy, handleROChange)}
            parentOnDrop={handleDrop}
            parentOnDragStart={handleDragStart}
            constraint={roConstraints(riHandle.x, cy)} />
        <DraggableHandle x={riHandle.x} y={riHandle.y}
            imageWidth={imageWidth}
            parentOnDrop={handleDrop}
            parentOnDragStart={handleDragStart}
            onDragMove={cbToRadius(cx, cy, handleRIChange)}
            constraint={riConstraint(roHandle.x, cy)} />
    </>);

    const frameViewWidgets = (
        <Ring cx={cx} cy={cy} ri={ri} ro={ro}
            imageWidth={imageWidth} />
    )

    const dispatch = useDispatch();

    const runAnalysis = () => {
        dispatch(analysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 1, {
            analysisType: AnalysisTypes.APPLY_RING_MASK,
            parameters: {
                shape: "ring",
                cx, cy, ri, ro,
            }
        }));
    };

    const {
        frameViewTitle, frameModeSelector,
        handles: resultHandles,
        widgets: resultWidgets,
    } = useDefaultFrameView({
        scanWidth,
        scanHeight,
        compoundAnalysisId: compoundAnalysis.compoundAnalysis,
        doAutoStart: compoundAnalysis.doAutoStart,
    })

    const subtitle = (
        <>{frameViewTitle} Ring: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), ri={ri.toFixed(2)}, ro={ro.toFixed(2)}</>
    )

    const toolbar = <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[1]} />

    return (
        <AnalysisLayoutTwoCol
            title="Ring analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles} extraWidgets={frameViewWidgets}
                    analysisIndex={0} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={imageWidth} height={imageHeight}
                    selectors={frameModeSelector}
                />
            </>}
            right={<>
                <ResultList
                    analysisIndex={1} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                    extraWidgets={resultWidgets}
                />
            </>}
            toolbar={toolbar}
        />
    );
}

export default RingMaskAnalysis;