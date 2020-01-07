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
import * as analysisActions from "../actions";
import { AnalysisProps } from "../types";
import AnalysisLayoutTwoRes from "./AnalysisLayoutTwoRes";
import useDefaultFrameView from "./DefaultFrameView";
import { useRectROI } from "./RectROI";
import Toolbar from "./Toolbar";




const ClustAnalysis: React.SFC<AnalysisProps> = ({ analysis, dataset }) => {

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

    const [delta, setDelta] = React.useState(0.05);

    const deltaChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setDelta(event.target.valueAsNumber);
    }

    const [minDist, setMinDist] = React.useState(1);

    const minDistChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setMinDist(event.target.valueAsNumber);
    }

    const [nPeaks, setNPeaks] = React.useState(50);

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
        dispatch(analysisActions.Actions.run(analysis.id, 1, {
            type: AnalysisTypes.SUM_SIG,
            parameters: {},
        }))
    }, [analysis.id, dispatch]);

    const runAnalysis = () => {
        dispatch(analysisActions.Actions.run(analysis.id, 2, {
            type: AnalysisTypes.CLUST,
            parameters: {
                roi: rectRoiParameters.roi,
                cx,
                cy,
                ri,
                ro,
                delta,
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
        analysisId: analysis.id,
    })

    const subtitle = (
        <>{frameViewTitle} Ring: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), ri={ri.toFixed(2)}, ro={ro.toFixed(2)}</>
    )
    const toolbar = <Toolbar analysis={analysis} onApply={runAnalysis} busyIdxs={[2]} />

    const [paramsVisible, setParamsVisible] = React.useState(false);

    const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
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
                        <label> Delta. Relative intensity difference for decision making for feature vector value (delta = (x-ref)/ref, so, normally, value should be in range [0,1]) <input type="number" value={delta} step="0.01" min="0" max="2" onChange={deltaChange} /> </label>
                    </Form.Field>
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
                    jobIndex={0} analysis={analysis.id}
                    width={imageWidth} height={imageHeight}
                    selectors={frameModeSelector}
                />
            </>}
            mid={<>
                <ResultList
                    jobIndex={1} analysis={analysis.id}
                    width={scanWidth} height={scanHeight}
                    extraHandles={rectRoiHandles}
                    extraWidgets={rectRoiWidgets}
                />
            </>}

            right={<>
                <ResultList
                    jobIndex={2} analysis={analysis.id}
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