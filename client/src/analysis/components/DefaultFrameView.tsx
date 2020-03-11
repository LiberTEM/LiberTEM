import * as React from "react";
import { useState } from "react";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import { useDiskROI } from "./DiskROI";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useRectROI } from "./RectROI";
import { useRoiPicker } from "./RoiPicker";


export enum DefaultModes {
    SUM = "SUM",
    SD = "SD",
    PICK = "PICK",
}

export enum DefaultRois {

    ALL = "ALL",
    DISK = "DISK",
    RECT = "RECT",

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
            text: "Standard Deviation",
            value: DefaultModes.SD,
        },
        {
            text: "Pick",
            value: DefaultModes.PICK,
        },
    ]

    const availableRois = [
        {
            text: "All",
            value: DefaultRois.ALL,
        },
        {
            text: "Disk",
            value: DefaultRois.DISK,
        },
        {
            text: "Rect",
            value: DefaultRois.RECT,
        },
    ]

    const [frameMode, setMode] = useState(DefaultModes.SUM);
    const [roi, setRoi] = useState(DefaultRois.ALL)

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={setMode} label="Mode" />

    let roiSelector = <ModeSelector modes={availableRois} currentMode={roi} onModeChange={setRoi} label="ROI" />

    if (frameMode === DefaultModes.PICK) {
        roiSelector = <></>;
    }

    const [cx, setCx] = React.useState(Math.round(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.round(scanHeight / 2));

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === DefaultModes.PICK,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        cx, cy, setCx, setCy
    });

    const { rectRoiHandles, rectRoiWidgets, rectRoiParameters } = useRectROI({ scanHeight, scanWidth })
    const { diskRoiHandles, diskRoiWidgets, diskRoiParameters } = useDiskROI({ scanHeight, scanWidth })

    const nullHandles: HandleRenderFunction = (onDragStart, onDrop) => null
    let handles = nullHandles;

    let widgets;
    let params = { roi: {} };
    switch (roi) {
        case DefaultRois.DISK:
            handles = diskRoiHandles;
            widgets = diskRoiWidgets;
            params = diskRoiParameters;
            break;
        case DefaultRois.RECT:
            handles = rectRoiHandles;
            widgets = rectRoiWidgets;
            params = rectRoiParameters;
            break;
    }

    switch (frameMode) {
        case DefaultModes.PICK:
            handles = pickHandles;
            widgets = undefined;
            break;
    }

    useRoiPicker({
        enabled: frameMode === DefaultModes.SD,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: params,
        analysis: AnalysisTypes.SD_FRAMES
    })

    useRoiPicker({
        enabled: frameMode === DefaultModes.SUM,
        scanWidth, scanHeight,
        jobIndex: 0,
        analysisId,
        roiParameters: params,
        analysis: AnalysisTypes.SUM_FRAMES,
    })

    const frameViewTitle = (
        frameMode !== DefaultModes.PICK ? null : <>Pick: x={pickCoords.cx}, y={pickCoords.cy} &emsp;</>
    )

    return {
        frameViewTitle,
        frameModeSelector: (<>{frameModeSelector} {roiSelector}</>),
        roiSelector,
        handles,
        widgets,
    }
}

export default useDefaultFrameView;