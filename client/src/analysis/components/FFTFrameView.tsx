import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import useFFTFramePicker from "./FFTFramePicker";
import useFFTSumFrames from "./FFTSumFrames";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRoiPicker } from "./RoiPicker";

const useFFTFrameView = ({
    scanWidth, scanHeight, analysisId, real_rad, real_centerx, real_centery
}: {
    scanWidth: number, scanHeight: number,
    analysisId: string, real_rad:number|null, real_centerx:number|null, real_centery:number|null
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

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={setMode} label="Mode"/>

    const [cx, setCx] = React.useState(Math.round(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.round(scanHeight / 2));

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 1,
        analysisId,
        cx, cy, setCx, setCy
    });
    
    useFFTFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        cx, cy, setCx, setCy,real_rad, real_centerx, real_centery
    });
    

    useRoiPicker({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 1,
        analysisId,
        scanWidth, scanHeight,
        roiParameters: {roi:{}},
        analysis: AnalysisTypes.SUM_FRAMES,
    })
    useFFTSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 0,
        analysisId,
        real_rad,
        real_centerx,
        real_centery
    })

    const frameViewTitle = (
        frameMode !== AnalysisTypes.PICK_FRAME ? null : <>Pick: x={pickCoords.cx}, y={pickCoords.cy} &emsp;</>
    )



    const nullHandles: HandleRenderFunction = (onDragStart, onDrop) => null

    return {
        frameViewTitle,
        handles: frameMode !== AnalysisTypes.PICK_FRAME ? nullHandles : pickHandles,

        frameModeSelector,
    }
}

export default useFFTFrameView;