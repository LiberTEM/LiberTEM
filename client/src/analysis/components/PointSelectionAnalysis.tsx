import * as React from "react";
import { useState } from "react";
import { connect } from "react-redux";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes, DatasetOpen } from "../../messages";
import { inRectConstraint } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
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

const PointSelectionAnalysis: React.SFC<MergedProps> = ({ analysis, dataset, run }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;

    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });

    const frameViewHandles: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={cx} y={cy} withCross={true}
            onDragMove={handleCenterChange}
            imageWidth={imageWidth}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
    </>);

    const { frameViewTitle, frameModeSelector, handles: resultHandles } = useDefaultFrameView({
        scanWidth,
        scanHeight,
        analysisId: analysis.id,
        run
    })

    const subtitle = (
        <>{frameViewTitle} Point: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)})</>
    )

    const runAnalysis = () => {
        run(analysis.id, 1, {
            type: AnalysisTypes.APPLY_POINT_SELECTOR,
            parameters: {
                shape: "point",
                cx,
                cy,
            }
        });
    };

    const toolbar = <Toolbar analysis={analysis} onApply={runAnalysis} busyIdxs={[1]} />

    return (
        <AnalysisLayoutTwoCol
            title="Point analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles}
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
                />
            </>}
            toolbar={toolbar}
        />
    );
}

export default connect(null, mapDispatchToProps)(PointSelectionAnalysis);