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
import * as analysisActions from "../actions";
import { AnalysisProps } from "../types";
import AnalysisLayoutThreeCol from "./AnalysisLayoutThreeCol";
import useFFTFrameView from "./FFTFrameView";
import Toolbar from "./Toolbar";




const FFTAnalysis: React.SFC<AnalysisProps> = ({ analysis, dataset }) => {
    const { shape } = dataset.params;
    const [scanHeight, scanWidth, imageHeight, imageWidth] = shape;
    const minLength = Math.min(imageWidth, imageHeight);

    const cx = imageWidth / 2;
    const cy = imageHeight / 2;
    const [rad_in, setRi] = useState(minLength / 4);
    const [rad_out, setRo] = useState(minLength / 2);

    const dispatch = useDispatch();
    const riHandle = {
        x: cx - rad_in,
        y: cy,
    }
    const roHandle = {
        x: cx - rad_out,
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
        <Ring cx={cx} cy={cy} ri={rad_in} ro={rad_out}
            imageWidth={imageWidth} />
    )

    const [check, setCheck] = React.useState(true);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setCheck(event.target.checked);
    }



    const [real_centerx, setCx] = useState(imageWidth / 2);
    const [real_centery, setCy] = useState(imageHeight / 2);
    const [real_rad, setR] = useState(minLength / 4);

    const handleCenterChange = defaultDebounce((newCx: number, newCy: number) => {
        setCx(newCx);
        setCy(newCy);
    });
    const handleRChange = defaultDebounce(setR);

    const rHandle = {
        x: real_centerx - real_rad,
        y: real_centery,
    }

    const frameViewHandlesreal: HandleRenderFunction = (handleDragStart, handleDrop) => (<>
        <DraggableHandle x={real_centerx} y={real_centery}
            imageWidth={imageWidth}
            onDragMove={handleCenterChange}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={inRectConstraint(imageWidth, imageHeight)} />
        <DraggableHandle x={rHandle.x} y={rHandle.y}
            imageWidth={imageWidth}
            onDragMove={cbToRadius(real_centerx, real_centery, handleRChange)}
            parentOnDragStart={handleDragStart}
            parentOnDrop={handleDrop}
            constraint={keepOnCY(real_centery)} />
    </>);

    const frameViewWidgetsreal = (
        <Disk cx={real_centerx} cy={real_centery} r={real_rad}
            imageWidth={imageWidth} imageHeight={imageHeight}
        />
    );

    const runAnalysis = () => {
        dispatch(analysisActions.Actions.run(analysis.id, 2, {
            type: AnalysisTypes.APPLY_FFT_MASK,
            parameters: {
                rad_in,
                rad_out,
                real_rad: check ? real_rad : null,
                real_centerx: check ? real_centerx : null,
                real_centery: check ? real_centery : null

            }
        }));
    };

    const { frameViewTitle, frameModeSelector, handles: resultHandles } = useFFTFrameView({
        scanWidth,
        scanHeight,
        analysisId: analysis.id,
        real_rad: check ? real_rad : null,
        real_centerx: check ? real_centerx : null,
        real_centery: check ? real_centery : null
    })



    const toolbar = <Toolbar analysis={analysis} onApply={runAnalysis} busyIdxs={[2]} />


    let subtitle;
    let mid: React.ReactNode;
    if (check) {
        mid = (<>
            <ResultList
                extraHandles={frameViewHandlesreal} extraWidgets={frameViewWidgetsreal}
                jobIndex={1} analysis={analysis.id}
                width={imageWidth} height={imageHeight}
                selectors={frameModeSelector}
            />
        </>)
        subtitle = (
            <>{frameViewTitle} real_rad={rad_in.toFixed(2)}, real_center=(x={real_centerx.toFixed(2)}, y={real_centery.toFixed(2)}), fourier_rad_in={rad_in.toFixed(2)}, fourier_rad_out={rad_out.toFixed(2)}</>
        )
    }

    else {
        mid = (<>
            <ResultList
                jobIndex={1} analysis={analysis.id}
                width={imageWidth} height={imageHeight}
                selectors={frameModeSelector}
            />
        </>)
        subtitle = (
            <>{frameViewTitle} fourier_rad_in={rad_in.toFixed(2)}, fourier_rad_out={rad_out.toFixed(2)}</>
        )
    }


    return (
        <AnalysisLayoutThreeCol
            title="FFT analysis" subtitle={subtitle}
            left={<>
                <ResultList
                    extraHandles={frameViewHandlesfft} extraWidgets={frameViewWidgetsfft}
                    jobIndex={0} analysis={analysis.id}
                    width={imageWidth} height={imageHeight}
                />
            </>}
            mid={mid}

            right={<>
                <ResultList
                    jobIndex={2} analysis={analysis.id}
                    width={scanWidth} height={scanHeight}
                    extraHandles={resultHandles}
                />
            </>}
            toolbar={toolbar}

            title2={<><label> Masking out of zero order diffraction peak <input type="checkbox" name="check" onChange={handleChange} checked={check} /> </label>
            </>}
            title1="Masking of intergation region in Fourier space"
            title3="Result of analysis"

        />
    );
}


export default FFTAnalysis;