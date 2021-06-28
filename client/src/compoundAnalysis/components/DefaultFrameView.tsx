import * as React from "react";
import { useState } from "react";
import { useDispatch } from "react-redux";
import { AnalysisTypes } from "../../messages";
import { HandleRenderFunction } from "../../widgets/types";
import * as compoundAnalysisActions from "../actions";
import useFramePicker from "./FramePicker";
import ModeSelector from "./ModeSelector";
import { useDiskROI } from "./roi/DiskROI";
import { useRectROI } from "./roi/RectROI";
import { useRoiPicker } from "./roi/RoiPicker";


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
    scanWidth, scanHeight, compoundAnalysisId, doAutoStart,
}: {
    scanWidth: number, scanHeight: number, compoundAnalysisId: string,
    doAutoStart: boolean,
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
    const [roi, setRoi] = useState(DefaultRois.ALL);

    const dispatch = useDispatch();

    const updateFrameMode = (newMode: string) => {
        dispatch(compoundAnalysisActions.Actions.enableAutoStart(compoundAnalysisId));
        setMode(newMode as DefaultModes);
    }

    const updateRoi = (newRoi: string) => {
        dispatch(compoundAnalysisActions.Actions.enableAutoStart(compoundAnalysisId));
        setRoi(newRoi as DefaultRois);
    }

    const frameModeSelector = <ModeSelector modes={availableModes} currentMode={frameMode} onModeChange={updateFrameMode} label="Mode" />

    let roiSelector = <ModeSelector modes={availableRois} currentMode={roi} onModeChange={updateRoi} label="ROI" />

    if (frameMode === DefaultModes.PICK) {
        roiSelector = <></>;
    }

    const [cx, setCx] = React.useState(Math.floor(scanWidth / 2));
    const [cy, setCy] = React.useState(Math.floor(scanHeight / 2));

    const { coords: pickCoords, handles: pickHandles } = useFramePicker({
        enabled: frameMode === DefaultModes.PICK,
        scanWidth, scanHeight,
        analysisIndex: 0,
        compoundAnalysisId,
        cx, cy, setCx, setCy
    });

    const { rectRoiHandles, rectRoiWidgets, rectRoiParameters } = useRectROI({ scanHeight, scanWidth })
    const { diskRoiHandles, diskRoiWidgets, diskRoiParameters } = useDiskROI({ scanHeight, scanWidth })

    const nullHandles: HandleRenderFunction = () => null
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
        enabled: frameMode === DefaultModes.SD && doAutoStart,
        scanWidth, scanHeight,
        analysisIndex: 0,
        compoundAnalysisId,
        roiParameters: params,
        analysisType: AnalysisTypes.SD_FRAMES
    })

    useRoiPicker({
        enabled: frameMode === DefaultModes.SUM && doAutoStart,
        scanWidth, scanHeight,
        analysisIndex: 0,
        compoundAnalysisId,
        roiParameters: params,
        analysisType: AnalysisTypes.SUM_FRAMES,
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
