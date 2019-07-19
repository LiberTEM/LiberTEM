import * as React from "react";
import { useState } from "react";
import { HandleRenderFunction } from "../../widgets/types";
import { useDiskROI } from "./DiskROI";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRectROI } from "./RectROI";
import { useRoiSumPicker } from "./RoiPicker";
import { useRoiSDPicker } from "./RoiSDPicker";
import useSDFrames from "./SDFrames";
import useSumFrames from "./SumFrames";

export enum DefaultModes {
    SUM = "SUM",
    SD = "SD",
    PICK = "PICK",
    SUM_DISK = "SUM_DISK",
    SD_RECT = "SD_RECT",
    SD_DISK ="SD_DISK",
}

const useDefaultFrameView = ({
    scanWidth, scanHeight, analysisId,
}: {
    scanWidth: number, scanHeight: number, analysisId: string,
}) => {
    /*const availableModes = [
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

    ];*/
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
            text: "SD over ROI (rect)",
            value: DefaultModes.SD_RECT,
        },
        {
            text: "SD over ROI (disk)",
            value: DefaultModes.SD_DISK,
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

    const { sumRoiHandles, sumRoiWidgets } = useRoiSumPicker({
        enabled: frameMode === DefaultModes.SUM_DISK,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
    });

    const { RectRoiHandles, RectRoiWidgets, RectroiParameters }  = useRectROI({ scanHeight, scanWidth })
    const { diskRoiHandles, diskRoiWidgets, diskroiParameters}  = useDiskROI({ scanHeight, scanWidth })

    useRoiSDPicker({
        enabled: frameMode === DefaultModes.SD_RECT,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: RectroiParameters,
        shapes: "rect"
    })

    useRoiSDPicker({
        enabled: frameMode === DefaultModes.SD_DISK,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: diskroiParameters,
        shapes: "disk"
    })

    // (frameMode === DefaultModes.SUM) ? true_expr : false_expr

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
            handles = sumRoiHandles;
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
            widgets = sumRoiWidgets;
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