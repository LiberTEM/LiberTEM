import { AnalysisDetails, AnalysisTypes } from "../messages";

export interface Analysis {
    id: string,
    dataset: string,
    currentJob: string | "",
    details: AnalysisDetails,
}

export type AnalysisState = Analysis;

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
}