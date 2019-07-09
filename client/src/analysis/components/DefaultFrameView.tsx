import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRoiSumPicker } from "./RoiPicker";
import { useRoiSDPicker } from "./RoiSDPicker";
import useSDFrames from "./SDFrames";
import useSumFrames from "./SumFrames";

const useDefaultFrameView = ({
    scanWidth, scanHeight, analysisId,
}: {
    scanWidth: number, scanHeight: number, analysisId: string,
}) => {
    const availableModes = [
        {
            text: "Average",
            value: AnalysisTypes.SUM_FRAMES,
        },

        {
            text: "SD",
            value: AnalysisTypes.SD_FRAMES,
        },

        {
            text: "Pick",
            value: AnalysisTypes.PICK_FRAME,
        },
        {
            text: "Average over ROI (disk)",
            value: AnalysisTypes.SUM_FRAMES_ROI,
        },
        {
            text: "SD over ROI (disk)",
            value: AnalysisTypes.SD_FRAMES_ROI,
        },

    ];

    const [frameMode, setMode] = useState(AnalysisTypes.SUM_FRAMES);

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={setMode} />

    const [cx, setCx] = React.useState(Math.round(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.round(scanHeight / 2));

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === AnalysisTypes.PICK_FRAME,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        cx, cy, setCx, setCy
    });

    const { sumRoiHandles, sumRoiWidgets } = useRoiSumPicker({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES_ROI,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
    });

    const { SDRoiHandles, SDRoiWidgets } = useRoiSDPicker({
        enabled: frameMode === AnalysisTypes.SD_FRAMES_ROI,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
    })

    useSumFrames({
        enabled: frameMode === AnalysisTypes.SUM_FRAMES,
        jobIndex: 0,
        analysisId,
    })

    useSDFrames({
        enabled: frameMode === AnalysisTypes.SD_FRAMES,
        jobIndex: 0,
        analysisId,
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
        case AnalysisTypes.SD_FRAMES_ROI:
            handles = SDRoiHandles;
            break;
}

    let widgets;

    switch (frameMode) {
        case AnalysisTypes.SUM_FRAMES_ROI:
            widgets = sumRoiWidgets;
            break;
        case AnalysisTypes.SD_FRAMES_ROI:
            widgets = SDRoiWidgets;
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