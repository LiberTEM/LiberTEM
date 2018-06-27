import { CenterOfMassParams, MaskDefDisk, MaskDefRing } from "../messages";

export enum AnalysisTypes {
    APPLY_RING_MASK = "APPLY_RING_MASK",
    APPLY_DISK_MASK = "APPLY_DISK_MASK",
    CENTER_OF_MASS = "CENTER_OF_MASS",
}

export interface RingMaskDetails {
    type: AnalysisTypes.APPLY_RING_MASK,
    parameters: MaskDefRing,
}

export interface DiskMaskDetails {
    type: AnalysisTypes.APPLY_DISK_MASK,
    parameters: MaskDefDisk,
}

export interface CenterOfMassDetails {
    type: AnalysisTypes.CENTER_OF_MASS,
    parameters: CenterOfMassParams,
}

export type AnalysisParameters = MaskDefRing | MaskDefDisk | CenterOfMassParams;
export type AnalysisDetails = RingMaskDetails | DiskMaskDetails | CenterOfMassDetails;

export interface Analysis {
    id: string,
    dataset: string,
    currentJob: string | "",
    details: AnalysisDetails,
}

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