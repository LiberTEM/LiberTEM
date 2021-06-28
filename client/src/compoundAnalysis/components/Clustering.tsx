import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { Accordion, Form, Icon } from "semantic-ui-react";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { cbToRadius, inRectConstraint, riConstraint, roConstraints } from "../../widgets/constraints";
import DraggableHandle from "../../widgets/DraggableHandle";
import Ring from "../../widgets/Ring";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import { CompoundAnalysisProps } from "../types";
import useDefaultFrameView from "./DefaultFrameView";
import AnalysisLayoutTwoRes from "./layouts/AnalysisLayoutTwoRes";
import { useRectROI } from "./roi/RectROI";
import Toolbar from "./Toolbar";


const ClustAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset }) => {
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

    const [minDist, setMinDist] = React.useState(1);

    const minDistChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setMinDist(event.target.valueAsNumber);
    }

    const [nPeaks, setNPeaks] = React.useState(500);

    const peakChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setNPeaks(event.target.valueAsNumber);
    }

    const [nClust, setNClust] = React.useState(20);

    const clustChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setNClust(event.target.valueAsNumber);
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
    const { rectRoiParameters, rectRoiHandles, rectRoiWidgets } = useRectROI({ scanWidth, scanHeight });

    React.useEffect(() => {
        if (compoundAnalysis.doAutoStart) {
            dispatch(compoundAnalysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 1, {
                analysisType: AnalysisTypes.SUM_SIG,
                parameters: {},
            }))
        }
    }, [compoundAnalysis.compoundAnalysis, dispatch, compoundAnalysis.doAutoStart]);

    const runAnalysis = () => {
        dispatch(compoundAnalysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 2, {
            analysisType: AnalysisTypes.CLUST,
            parameters: {
                roi: rectRoiParameters.roi,
                cx,
                cy,
                ri,
                ro,
                n_clust: nClust,
                n_peaks: nPeaks,
                min_dist: minDist
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
    const toolbar = <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[2]} />

    const [paramsVisible, setParamsVisible] = React.useState(false);

    const handleClick = () => {
        setParamsVisible(!paramsVisible);
    }

    const clustparams =
        <Accordion>
            <Accordion.Title active={paramsVisible} index={0} onClick={handleClick}>
                <Icon name='dropdown' />
                Parameters
            </Accordion.Title>
            <Accordion.Content active={paramsVisible}>
                <Form>
                    <Form.Field>
                        <label> Number of clusters  <input type="number" value={nClust} step="1" min="2" max="100" onChange={clustChange} /> </label>
                    </Form.Field>
                    <Form.Field>
                        <label>  Maximal number of possible peak positions to detect (better put higher value,
        the output is limited to the number of peaks the algorithm could find)  <input type="number" value={nPeaks} step="1" min="5" max="200" onChange={peakChange} /> </label>
                    </Form.Field>
                    <Form.Field>
                        <label>  Minimal distance in pixels between peaks  <input type="number" value={minDist} step="1" min="0" max="100" onChange={minDistChange} />  </label>
                    </Form.Field>
                </Form>
            </Accordion.Content>
        </Accordion>
    return (
        <AnalysisLayoutTwoRes
            title="Region clustering" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandles} extraWidgets={frameViewWidgets}
                    analysisIndex={0} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={imageWidth} height={imageHeight}
                    selectors={frameModeSelector}
                />
            </>}
            mid={<>
                <ResultList
                    analysisIndex={1} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={scanWidth} height={scanHeight}
                    extraHandles={rectRoiHandles}
                    extraWidgets={rectRoiWidgets}
                />
            </>}

            right={<>
                <ResultList
                    analysisIndex={2} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                    extraWidgets={resultWidgets}
                />
            </>}
            toolbar={toolbar}
            clustparams={clustparams}

            title1="Peaks inside the ring will be considered"
            title2="Choose specimen region"
            title3="Clustering result"

        />
    );
}

export default ClustAnalysis;