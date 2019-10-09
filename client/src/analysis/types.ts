import { AnalysisTypes, DatasetOpen, PickFrameDetails, SumFramesDetails } from "../messages";
import CenterOfMassAnalysis from "./components/CenterOfMassAnalysis";
import ClustAnalysis from "./components/Clustering";
import DiskMaskAnalysis from "./components/DiskMaskAnalysis";
import FEM from "./components/FEM";
import FFTAnalysis from "./components/FFTAnalysis";
import PointSelectionAnalysis from "./components/PointSelectionAnalysis";
import RadialFourierAnalysis from "./components/RadialFourierAnalysis";
import RingMaskAnalysis from "./components/RingMaskAnalysis";



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


/**
 * AnalysisProps is the interface you should implement with your Analysis, as a functional component.
 * For example:
 * 
 * const MyAnalysis: React.FunctionalComponent<AnalysisProps> = ({ analysis, dataset }) = { ... }
 * 
 */
export interface AnalysisProps {
    analysis: AnalysisState,
    dataset: DatasetOpen,
}

export type AnalysisState = Analysis & {
};

export interface AnalysisMetadataItem {
    desc: string,
    title: string,
    component?: React.FunctionComponent<AnalysisProps>,
}

/**
 * list of all analyses; those having a component will be available for selection in the UI
 * 
 * please fill in a title and description, and reference your component.
 */
export const AnalysisMetadata: { [s: string]: AnalysisMetadataItem } = {
    [AnalysisTypes.APPLY_RING_MASK]: {
        desc: "Apply a ring mask with center cx, cy; inner radius ri, outer radius ro",
        title: "Ring",
        component: RingMaskAnalysis,
    },
    [AnalysisTypes.APPLY_DISK_MASK]: {
        desc: "Apply a disk mask with center cx, cy; radius r",
        title: "Disk",
        component: DiskMaskAnalysis,
    },
    [AnalysisTypes.FEM]: {
        desc: "Apply a ring mask with center cx, cy; radius r",
        title: "FEM (SD over Ring)",
        component: FEM,
    },
    [AnalysisTypes.CENTER_OF_MASS]: {
        desc: "Compute the center of mass of all diffraction images",
        title: "Center of mass",
        component: CenterOfMassAnalysis,
    },
    [AnalysisTypes.APPLY_POINT_SELECTOR]: {
        desc: "Create an image from a single pixel selected in the detector",
        title: "Point selection",
        component: PointSelectionAnalysis,
    },
    [AnalysisTypes.SUM_FRAMES]: {
        desc: "Create a sum of all detector frames",
        title: "Sum all frames",
    },
    [AnalysisTypes.SD_FRAMES]: {
        desc: "Create a SD of all detector frames",
        title: "SD all frames",
    },
    [AnalysisTypes.PICK_FRAME]: {
        desc: "Pick a single frame",
        title: "Pick frame",
    },
    [AnalysisTypes.PICK_FFT_FRAME]: {
        desc: "Pick a single frame",
        title: "Pick frame",
    },
    [AnalysisTypes.APPLY_FFT_MASK]: {
        desc: "Apply a ring mask with inner radius fourier_rad_in and outer radius fourier_rad_out in Fourier space",
        title: "FFT analysis",
        component: FFTAnalysis
    },
    [AnalysisTypes.FFTSUM_FRAMES]: {
        desc: "Fourier transform of sum of all detector frames",
        title: "FFT of Sum all frames",
    },
    [AnalysisTypes.RADIAL_FOURIER]: {
        desc: "Compute a radial Fourier analysis",
        title: "Radial Fourier",
        component: RadialFourierAnalysis,
    },
    [AnalysisTypes.CLUST]: {
        desc: "Region clustering based on non-zero order diffraction peaks positions",
        title: "Clustering",
        component: ClustAnalysis,
    },
    [AnalysisTypes.SUM_SIG]: {
        desc: "Frame integration",
        title: "Sum",
    },
}