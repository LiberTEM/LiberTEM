import { AnalysisDetails, AnalysisTypes } from "../messages";

export interface Analysis {
    id: string,
    dataset: string,
    currentJob: string | "",
    details: AnalysisDetails,
}

export type AnalysisState = Analysis;

export const AnalysisMetadata: { [s: string]: { [s: string]: string } } = {
    [AnalysisTypes.APPLY_RING_MASK]: {
        long: "Apply a ring mask with center cx, cy; inner radius ri, outer radius ro",
        short: "Ring",
    },
    [AnalysisTypes.APPLY_DISK_MASK]: {
        "long": "Apply a disk mask with center cx, cy; radius r",
        "short": "Disk",
    },
    [AnalysisTypes.CENTER_OF_MASS]: {
        "long": "Compute the center of mass of all diffraction images",
        "short": "Center of mass",
    },
}