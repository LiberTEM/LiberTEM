import * as React from "react";
import { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Accordion, Checkbox, Form, Icon, Input } from "semantic-ui-react";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { RootReducer } from "../../store";
import { cbToRadius, inRectConstraint, keepOnCY } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import { DraggableHandle } from "../../widgets/DraggableHandle";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import { haveDisplayResult } from "../helpers";
import { CompoundAnalysisProps } from "../types";
import useDefaultFrameView from "./DefaultFrameView";
import AnalysisLayoutTwoCol from "./layouts/AnalysisLayoutTwoCol";
import Toolbar from "./Toolbar";

const CenterOfMassAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);
    const [cx, setCx] = useState(imageWidth / 2);
    const [cy, setCy] = useState(imageHeight / 2);
    const [r, setR] = useState(minLength / 4);
    const [flip_y, setFlipY] = useState(false);
    const [scan_rotation, setScanRotation] = useState(0.0);
    const [paramsVisible, setParamsVisible] = useState(false);

    const dispatch = useDispatch();

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
        compoundAnalysisId: compoundAnalysis.compoundAnalysis,
        doAutoStart: compoundAnalysis.doAutoStart,
    })

    const subtitle = <>{frameViewTitle} Disk: center=(x={cx.toFixed(2)}, y={cy.toFixed(2)}), r={r.toFixed(2)}</>;

    const runAnalysis = () => {
        dispatch(compoundAnalysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 1, {
            analysisType: AnalysisTypes.CENTER_OF_MASS,
            parameters: {
                shape: "com",
                cx,
                cy,
                r,
                flip_y,
                scan_rotation,
            }
        }));
    };

    const toolbar = <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[1]} />

    const toggleParamsVisible = () => setParamsVisible(!paramsVisible);
    const updateFlipY = (e: React.ChangeEvent<HTMLInputElement>, { checked }: { checked: boolean }) => {
        setFlipY(checked);
    };
    const updateScanRotation = (e: React.ChangeEvent<HTMLInputElement>, { value }: { value: string }) => {
        let newScanRotation = parseFloat(value);
        if (!newScanRotation) {
            newScanRotation = 0.0;
        }
        setScanRotation(newScanRotation);
    };

    const analyses = useSelector((state: RootReducer) => state.analyses)
    const jobs = useSelector((state: RootReducer) => state.jobs)

    const haveResult = haveDisplayResult(
        compoundAnalysis,
        analyses,
        jobs,
        [1],
    );

    React.useEffect(() => {
        if (haveResult) {
            runAnalysis();
        }
    }, [haveResult, flip_y, scan_rotation]);

    const comParams = (
        <Accordion>
            <Accordion.Title active={paramsVisible} index={0} onClick={toggleParamsVisible}>
                <Icon name='dropdown' />
                Parameters
            </Accordion.Title>
            <Accordion.Content active={paramsVisible}>
                <Form>
                    <Form.Field control={Checkbox} label="Flip in y direction" checked={flip_y} onChange={updateFlipY} />
                    <Form.Field type="number" control={Input} label="Rotation between scan and detector (deg)" value={scan_rotation} onChange={updateScanRotation} />
                </Form>
            </Accordion.Content>
        </Accordion>
    );

    return (
        <AnalysisLayoutTwoCol
            title="COM analysis" subtitle={subtitle}
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
            params={comParams}
        />
    );
}

export default CenterOfMassAnalysis;
