import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import useFFTFramePicker from "./FFTFramePicker";
import useFFTSumFrames from "./FFTSumFrames";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRoiPicker } from "./roi/RoiPicker";

const useFFTFrameView = ({
    scanWidth, scanHeight, compoundAnalysisId, real_rad, real_centerx,
    real_centery, doAutoStart,
}: {
    scanWidth: number, scanHeight: number, compoundAnalysisId: string,
    real_rad: number | null, real_centerx: number | null, real_centery: number | null,
    doAutoStart: boolean,
}) => {
    const availableModes = [
        {
            text: "Pick",
            value: AnalysisTypes.PICK_FRAME,
        },
        {
            text: "Average",
            value: AnalysisTypes.SUM_FRAMES,
        },
    ];

    const [frameMode, setMode] = useState(AnalysisTypes.SUM_FRAMES);

    const dispatch = useDispatch();

    const updateMode = (newMode: string) => {
        dispatch(compoundAnalysisActions.Actions.enableAutoStart(compoundAnalysisId));
        setMode(newMode as AnalysisTypes);
    }

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={updateMode} label="Mode" />

    const [cx, setCx] = React.useState(Math.floor(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.floor(scanHeight / 2));

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME && doAutoStart,
        scanWidth, scanHeight,
        analysisIndex: 1,
        compoundAnalysisId,
        cx, cy, setCx, setCy
    });

    useFFTFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME && doAutoStart,
        scanWidth, scanHeight,
        analysisIndex: 0,
        compoundAnalysisId,
        cx, cy, setCx, setCy, real_rad, real_centerx, real_centery
    });

    useRoiPicker({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES && doAutoStart,
        analysisIndex: 1,
        compoundAnalysisId,
        scanWidth, scanHeight,
        roiParameters: { roi: {} },
        analysisType: AnalysisTypes.SUM_FRAMES,
    })
    useFFTSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES && doAutoStart,
        analysisIndex: 0,
        compoundAnalysisId,
        real_rad,
        real_centerx,
        real_centery
    })

    const frameViewTitle = (
        frameMode !== AnalysisTypes.PICK_FRAME ? null : <>Pick: x={pickCoords.cx}, y={pickCoords.cy} &emsp;</>
    )

    const nullHandles: HandleRenderFunction = () => null

    return {
        frameViewTitle,
        handles: frameMode !== AnalysisTypes.PICK_FRAME ? nullHandles : pickHandles,

        frameModeSelector,
    }
}

export default useFFTFrameView;
