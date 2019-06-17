import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import * as analysisActions from "../actions";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRoiPicker } from "./RoiPicker";
import useSumFrames from "./SumFrames";

const useDefaultFrameView = ({
    scanWidth, scanHeight, analysisId, run
}: {
    scanWidth: number, scanHeight: number,
    analysisId: string, run: typeof analysisActions.Actions.run
}) => {
    const availableModes = [
        {
            text: "Average",
            value: AnalysisTypes.SUM_FRAMES,
        },
        {
            text: "Pick",
            value: AnalysisTypes.PICK_FRAME,
        },
        {
            text: "Average over ROI (disk)",
            value: AnalysisTypes.SUM_FRAMES_ROI,
        }
    ];

    const [frameMode, setMode] = useState(AnalysisTypes.SUM_FRAMES);

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={setMode} />

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        run
    });

    const { sumRoiHandles, sumRoiWidgets } = useRoiPicker({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES_ROI,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        run
    })

    useSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 0,
        analysisId,
        run
    })

    const frameViewTitle = (
        frameMode !== AnalysisTypes.PICK_FRAME ? null : <>Pick: x={pickCoords.cx}, y={pickCoords.cy} &emsp;</>
    )

    const nullHandles: HandleRenderFunction = (onDragStart, onDrop) => null

    let handles = nullHandles;

    switch (frameMode) {
        case AnalysisTypes.PICK_FRAME:
            handles = pickHandles;
            break;
        case AnalysisTypes.SUM_FRAMES_ROI:
            handles = sumRoiHandles;
            break;
    }

    let widgets;

    switch (frameMode) {
        case AnalysisTypes.SUM_FRAMES_ROI:
            widgets = sumRoiWidgets;
            break;
    }

    return {
        frameViewTitle,
        frameModeSelector,
        handles,
        widgets,
    }
}

export default useDefaultFrameView;