import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { defaultDebounce } from "../../helpers";
import ResultList from "../../job/components/ResultList";
import { AnalysisTypes } from "../../messages";
import { cbToRadius, inRectConstraint, keepOnCY, riConstraint, roConstraints } from "../../widgets/constraints";
import Disk from "../../widgets/Disk";
import DraggableHandle from "../../widgets/DraggableHandle";
import Ring from "../../widgets/Ring";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import { CompoundAnalysisProps } from "../types";
import useFFTFrameView from "./FFTFrameView";
import AnalysisLayoutThreeCol from "./layouts/AnalysisLayoutThreeCol";
import Toolbar from "./Toolbar";


const FFTAnalysis: React.FC<CompoundAnalysisProps> = ({ compoundAnalysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);

    const cx = imageWidth / 2;
    const cy = imageHeight / 2;
    const [radIn, setRi] = useState(minLength / 4);
    const [radOut, setRo] = useState(minLength / 2);

    const dispatch = useDispatch();
    const riHandle = {
        x: cx - radIn,
        y: cy,
    }
    const roHandle = {
        x: cx - radOut,
        y: cy,
    }


    const handleRIChange = defaultDebounce(setRi);
    const handleROChange = defaultDebounce(setRo);

    const frameViewHandlesfft: HandleRenderFunction = (handleDragStart, handleDrop) => (<>

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

    const frameViewWidgetsfft = (
        <Ring cx={cx} cy={cy} ri={radIn} ro={radOut}
            imageWidth={imageWidth} />
    )

    const [check, setCheck] = React.useState(true);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        dispatch(compoundAnalysisActions.Actions.enableAutoStart(compoundAnalysis.compoundAnalysis));
        setCheck(event.target.checked);
    }

    const [realCenterX, setCx] = useState(imageWidth / 2);
    const [realCenterY, setCy] = useState(imageHeight / 2);
    const [realRad, setR] = useState(minLength / 4);

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        dispatch(compoundAnalysisActions.Actions.enableAutoStart(compoundAnalysis.compoundAnalysis));
        setCx(newCx);
        setCy(newCy);
    });
    const handleRChange = defaultDebounce(setR);

    const rHandle = {
        x: realCenterX - realRad,
        y: realCenterY,
    }

    const frameViewHandlesreal: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={realCenterX} y={realCenterY}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(realCenterX, realCenterY, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={keepOnCY(realCenterY)} />
    </>);

    const frameViewWidgetsreal = (
        <Disk cx={realCenterX} cy={realCenterY} r={realRad} imageWidth={imageWidth} />
    );

    const runAnalysis = () => {
        dispatch(compoundAnalysisActions.Actions.run(compoundAnalysis.compoundAnalysis, 2, {
            analysisType: AnalysisTypes.APPLY_FFT_MASK,
            parameters: {
                rad_in: radIn,
                rad_out: radOut,
                real_rad: check ? realRad : null,
                real_centerx: check ? realCenterX : null,
                real_centery: check ? realCenterY : null
            }
        }));
    };

    const { frameViewTitle, frameModeSelector, handles: resultHandles } = useFFTFrameView({
        scanWidth,
        scanHeight,
        compoundAnalysisId: compoundAnalysis.compoundAnalysis,
        real_rad: check ? realRad : null,
        real_centerx: check ? realCenterX : null,
        real_centery: check ? realCenterY : null,
        doAutoStart: compoundAnalysis.doAutoStart,
    });

    const toolbar = <Toolbar compoundAnalysis={compoundAnalysis} onApply={runAnalysis} busyIdxs={[2]} />

    let subtitle;
    let mid: React.ReactNode;
    if (check) {
        mid = (<>
            <ResultList
                extraHandles={frameViewHandlesreal} extraWidgets={frameViewWidgetsreal}
                analysisIndex={1} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                width={imageWidth} height={imageHeight}
                selectors={frameModeSelector}
            />
        </>)
        subtitle = (
            <>{frameViewTitle} real_rad={radIn.toFixed(2)}, real_center=(x={realCenterX.toFixed(2)}, y={realCenterY.toFixed(2)}), fourier_rad_in={radIn.toFixed(2)}, fourier_rad_out={radOut.toFixed(2)}</>
        )
    }
    else {
        mid = (<>
            <ResultList
                analysisIndex={1} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                width={imageWidth} height={imageHeight}
                selectors={frameModeSelector}
            />
        </>)
        subtitle = (
            <>{frameViewTitle} fourier_rad_in={radIn.toFixed(2)}, fourier_rad_out={radOut.toFixed(2)}</>
        )
    }


    return (
        <AnalysisLayoutThreeCol
            title="FFT analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandlesfft} extraWidgets={frameViewWidgetsfft}
                    analysisIndex={0} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={imageWidth} height={imageHeight}
                />
            </>}
            mid={mid}

            right={<>
                <ResultList
                    analysisIndex={2} compoundAnalysis={compoundAnalysis.compoundAnalysis}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                />
            </>}
            toolbar={toolbar}

            title2={<><label> Masking out of zero order diffraction peak <input type="checkbox" name="check" onChange={handleChange} checked={check} /> </label>
            </>}
            title1="Masking of integration region in Fourier space"
            title3="Result of analysis"

        />
    );
}


export default FFTAnalysis;
