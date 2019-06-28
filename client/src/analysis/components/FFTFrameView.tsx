import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import useFFTSumFrames from "./FFTSumFrames";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import useSumFrames from "./SumFrames";

const useFFTFrameView = ({
    scanWidth, scanHeight, analysisId
}: {
    scanWidth: number, scanHeight: number,
    analysisId: string,
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

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={setMode} />

    /*useFrameFFTPicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
    });*/

    const [cx, setCx] = React.useState(Math.round(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.round(scanHeight / 2));

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 1,
        analysisId,
        cx, cy, setCx, setCy, type: AnalysisTypes.PICK_FRAME
    });
    
    useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        cx, cy, setCx, setCy, type: AnalysisTypes.PICK_FFT_FRAME
    });
    

    useSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 1,
        analysisId,
    })

    useFFTSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 0,
        analysisId,
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