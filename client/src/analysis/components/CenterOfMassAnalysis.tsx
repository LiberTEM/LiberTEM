import * as React from "react";
import { useState } from "react";
import { connect } from "react-redux";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes, DatasetOpen } from "../../messages";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import { DraggableHandle } from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import { AnalysisState } from "../types";
import AnalysisLayoutTwoCol from "./AnalysisLayoutTwoCol";
import useDefaultFrameView from "./DefaultFrameView";
import Toolbar from "./Toolbar";

interface AnalysisProps {
    analysis: AnalysisState,
    dataset: DatasetOpen,
}

const mapDispatchToProps = {
    run: analysisActions.Actions.run,
}

type MergedProps = AnalysisProps & DispatchProps<typeof mapDispatchToProps>

const CenterOfMassAnalysis: React.SFC<MergedProps> = ({ analysis, dataset, run }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);
    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const [r, setR] = useState(minLength / 4);

    const rHandle = {
        x: cx - r,
        y: cy,
    }

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });
    const handleRChange = defaultDebounce(setR);

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
            imageWidth={imageWidth} imageHeight={imageHeight} />
    )

    const {
        frameViewTitle, frameModeSelector,
        handles: resultHandles, widgets: resultWidgets
    } = useDefaultFrameView({
        scanWidth,
        scanHeight,
        analysisId: analysis.id,
        run
    })

    const subtitle = <>{frameViewTitle} Disk: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), r={r.toFixed(2)}</>;

    const runAnalysis = () => {
        run(analysis.id, 1, {
            type: AnalysisTypes.CENTER_OF_MASS,
            parameters: {
                shape: "com",
                cx,
                cy,
                r
            }
        });
    };

    const toolbar = <Toolbar analysis={analysis} onApply={runAnalysis} busyIdxs={[1]} />

    return (
        <AnalysisLayoutTwoCol
            title="COM analysis" subtitle={subtitle}
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

export default connect(null, mapDispatchToProps)(CenterOfMassAnalysis);