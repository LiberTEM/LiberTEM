import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import { useDiskROI } from "./DiskROI";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRectROI } from "./RectROI";
import { useRoiPicker } from "./RoiPicker";
import useSDFrames from "./SDFrames";
import useSumFrames from "./SumFrames";

export enum DefaultModes {
    SUM = "SUM",
    SD = "SD",
    PICK = "PICK",
    SUM_DISK = "SUM_DISK",
    SUM_RECT = "SUM_RECT",
    SD_RECT = "SD_RECT",
    SD_DISK ="SD_DISK",
}

const useDefaultFrameView = ({
    scanWidth, scanHeight, analysisId,
}: {
    scanWidth: number, scanHeight: number, analysisId: string,
}) => {
    const availableModes = [
        {
            text: "Average",
            value: DefaultModes.SUM,
        },
        {
            text: "SD",
            value: DefaultModes.SD,
        },
        {
            text: "PICK",
            value: DefaultModes.PICK,
        },
        {
            text: "Average over ROI (disk)",
            value: DefaultModes.SUM_DISK,
        },
        {
            text: "Average over ROI (rect)",
            value: DefaultModes.SUM_RECT,
        },
        {
            text: "SD over ROI (disk)",
            value: DefaultModes.SD_DISK,
        },
        {
            text: "SD over ROI (rect)",
            value: DefaultModes.SD_RECT,
        },
    ]

    const [frameMode, setMode] = useState(DefaultModes.SUM);

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={setMode} />

    const [cx, setCx] = React.useState(Math.round(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.round(scanHeight / 2));


    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === DefaultModes.PICK,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        cx, cy, setCx, setCy
    });


    const { RectRoiHandles, RectRoiWidgets, RectroiParameters }  = useRectROI({ scanHeight, scanWidth })
    const { diskRoiHandles, diskRoiWidgets, diskroiParameters}  = useDiskROI({ scanHeight, scanWidth })

    useRoiPicker({
        enabled: frameMode === DefaultModes.SD_RECT,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: RectroiParameters,
        analys: AnalysisTypes.SD_FRAMES
    })

    

    useRoiPicker({
        enabled: frameMode === DefaultModes.SD_DISK,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: diskroiParameters,
        analys: AnalysisTypes.SD_FRAMES,
    })

    useRoiPicker({
        enabled: frameMode === DefaultModes.SUM_DISK,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: diskroiParameters,
        analys: AnalysisTypes.SUM_FRAMES,
    })

    useRoiPicker({
        enabled: frameMode === DefaultModes.SUM_RECT,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: RectroiParameters,
        analys: AnalysisTypes.SUM_FRAMES,
    })

    useSumFrames({
        enabled: frameMode === DefaultModes.SUM,
        jobIndex: 0,
        analysisId,
    })

    useSDFrames({
        enabled: frameMode === DefaultModes.SD,
        jobIndex: 0,
        analysisId,
    })

    const frameViewTitle = (
        frameMode !== DefaultModes.PICK ? null : <>Pick: x={pickCoords.cx}, y={pickCoords.cy} &emsp;</>
    )

    const nullHandles: HandleRenderFunction = (onDragStart, onDrop) => null

    let handles = nullHandles;

    switch (frameMode) {
        case DefaultModes.PICK:
            handles = pickHandles;
            break;
        case DefaultModes.SUM_DISK:
            handles = diskRoiHandles;
            break;
        case DefaultModes.SUM_RECT:
            handles = RectRoiHandles;
            break;  
        case DefaultModes.SD_RECT:
            handles = RectRoiHandles;
            break;
        case DefaultModes.SD_DISK:
            handles = diskRoiHandles;
            break;
}

    let widgets;

    switch (frameMode) {
        case DefaultModes.SUM_DISK:
            widgets = diskRoiWidgets;
            break;
        case DefaultModes.SUM_RECT:
            widgets = RectRoiWidgets;
            break;
        case DefaultModes.SD_RECT:
            widgets = RectRoiWidgets;
            break;
        case DefaultModes.SD_DISK:
            widgets = diskRoiWidgets;
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