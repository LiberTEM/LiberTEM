import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import useFFTSumFrames from "./FFTSumFrames";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import useSumFrames from "./SumFrames";

const useFFTFrameView = ({
    scanWidth, scanHeight, analysisId, run
}: {
    scanWidth: number, scanHeight: number,
    analysisId: string, run: typeof analysisActions.Actions.run
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

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 1,
        analysisId,
        run
    });

    useSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 1,
        analysisId,
        run
    })

    useFFTSumFrames({
        enabled: true, // frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 0,
        analysisId,
        run
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