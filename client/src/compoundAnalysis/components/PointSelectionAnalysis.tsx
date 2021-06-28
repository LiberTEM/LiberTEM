import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { CompoundAnalysisProps } from "../types";
import useDefaultFrameView from "./DefaultFrameView";
import AnalysisLayoutTwoCol from "./layouts/AnalysisLayoutTwoCol";
import Toolbar from "./Toolbar";


const PointSelectionAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset, }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;

    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        const newX = Math.round(newCx);
        const newY = Math.round(newCy);
        if (cx === newX && cy === newY) {
            return;
        }
        setCx(newX);
        setCy(newY);
    });

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy} withCross
            onDragMove={handleCenterChange}
            imageWidth={imageWidth}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
    </>);

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
        <>{frameViewTitle} Point: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)})</>
    )

    const dispatch = useDispatch();

    const runAnalysis = () => {
        dispatch(analysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 1, {
            analysisType: AnalysisTypes.APPLY_POINT_SELECTOR,
            parameters: {
                shape: "point",
                cx,
                cy,
            }
        }));
    };

    const toolbar = <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[1]} />

    return (
        <AnalysisLayoutTwoCol
            title="Point analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles}
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

export default PointSelectionAnalysis;