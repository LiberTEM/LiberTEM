import { AnalysisTypes, PickFrameDetails, SumFramesDetails } from "../messages";

export type AnalysisStatus = "busy" | "idle";

export type JobList = string[];

export type JobHistory = string[][];

export type FrameAnalysisDetails = PickFrameDetails | SumFramesDetails;

export interface Analysis {
    id: string,
    dataset: string,
    jobs: JobList,
    jobHistory: JobHistory,
    mainAnalysisType: AnalysisTypes,
}

export type AnalysisState = Analysis & {
};

interface AnalysisMetadataItem {
    long: string,
    short: string,
    showInUI: boolean,
}

export const AnalysisMetadata: { [s: string]: AnalysisMetadataItem } = {
    [AnalysisTypes.APPLY_RING_MASK]: {
        long: "Apply a ring mask with center cx, cy; inner radius ri, outer radius ro",
        short: "Ring",
        showInUI: true,
    },
    [AnalysisTypes.APPLY_DISK_MASK]: {
        long: "Apply a disk mask with center cx, cy; radius r",
        short: "Disk",
        showInUI: true,
    },
    [AnalysisTypes.CENTER_OF_MASS]: {
        long: "Compute the center of mass of all diffraction images",
        short: "Center of mass",
        showInUI: true,
    },
    [AnalysisTypes.APPLY_POINT_SELECTOR]: {
        long: "Create an image from a single pixel selected in the detector",
        short: "Point selection",
        showInUI: true,
    },
    [AnalysisTypes.SUM_FRAMES]: {
        long: "Create a sum of all detector frames",
        short: "Sum all frames",
        showInUI: false,
    },
    [AnalysisTypes.SUM_FRAMES_ROI]: {
        long: "Create a sum of ROI",
        short: "Sum over ROI",
        showInUI: false,
    },
    [AnalysisTypes.PICK_FRAME]: {
        long: "Pick a single frame",
        short: "Pick frame",
        showInUI: false,
    },
    [AnalysisTypes.RADIAL_FOURIER]: {
        long: "Compute a radial Fourier analysis",
        short: "Radial Fourier",
        showInUI: true,
    },
}