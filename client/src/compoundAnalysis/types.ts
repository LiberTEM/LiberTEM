import { AnalysisTypes, DatasetOpen, PickFrameDetails, SumFramesDetails } from "../messages";
import CenterOfMassAnalysis from "./components/CenterOfMassAnalysis";
import ClustAnalysis from "./components/Clustering";
import DiskMaskAnalysis from "./components/DiskMaskAnalysis";
import FEM from "./components/FEM";
import FFTAnalysis from "./components/FFTAnalysis";
import PointSelectionAnalysis from "./components/PointSelectionAnalysis";
import RadialFourierAnalysis from "./components/RadialFourierAnalysis";
import RingMaskAnalysis from "./components/RingMaskAnalysis";


export type CompoundAnalysisStatus = "busy" | "idle";

export type JobHistory = string[][];

export type FrameAnalysisDetails = PickFrameDetails | SumFramesDetails;

export interface CompoundAnalysis {
    compoundAnalysis: string,
    dataset: string,
    details: {
        mainType: AnalysisTypes,
        analyses: string[],
    }
}


/**
 * CompoundAnalysisProps is the interface you should implement with your
 * CompoundAnalysis, as a functional component. For example:
 * 
 * const MyAnalysis: React.FunctionalComponent<CompoundAnalysisProps> = ({ analysis, dataset }) = { ... }
 * 
 */

export type CompoundAnalysisState = CompoundAnalysis & {
    doAutoStart: boolean,
};

export interface CompoundAnalysisProps {
    compoundAnalysis: CompoundAnalysisState,
    dataset: DatasetOpen,
}

export interface CompoundAnalysisMetadataItem {
    desc: string,
    title: string,
    component?: React.FunctionComponent<CompoundAnalysisProps>,
}


/**
 * list of all analyses; those having a component will be available for selection in the UI
 * 
 * please fill in a title and description, and reference your component.
 */
export const CompoundAnalysisMetadata: { [s: string]: CompoundAnalysisMetadataItem } = {
    [AnalysisTypes.APPLY_RING_MASK]: {
        desc: "Application of a ring mask",
        title: "Ring",
        component: RingMaskAnalysis,
    },
    [AnalysisTypes.APPLY_DISK_MASK]: {
        desc: "Application of a disk mask",
        title: "Disk",
        component: DiskMaskAnalysis,
    },
    [AnalysisTypes.FEM]: {
        desc: "Application of a ring mask",
        title: "FEM (SD over Ring)",
        component: FEM,
    },
    [AnalysisTypes.CENTER_OF_MASS]: {
        desc: "The center of mass of all diffraction images",
        title: "Center of mass",
        component: CenterOfMassAnalysis,
    },
    [AnalysisTypes.APPLY_POINT_SELECTOR]: {
        desc: "Image from a single pixel selected in the detector",
        title: "Point selection",
        component: PointSelectionAnalysis,
    },
    [AnalysisTypes.SUM_FRAMES]: {
        desc: "A sum of all (or selected) detector frames",
        title: "Sum all frames",
    },
    [AnalysisTypes.SD_FRAMES]: {
        desc: "A SD of all (or selected) detector frames",
        title: "SD all frames",
    },
    [AnalysisTypes.PICK_FRAME]: {
        desc: "A single picked frame",
        title: "Pick frame",
    },
    [AnalysisTypes.PICK_FFT_FRAME]: {
        desc: "Fourier transform of a picked frame",
        title: "Pick frame",
    },
    [AnalysisTypes.APPLY_FFT_MASK]: {
        desc: "Application of a ring mask in Fourier space",
        title: "FFT analysis",
        component: FFTAnalysis
    },
    [AnalysisTypes.FFTSUM_FRAMES]: {
        desc: "Fourier transform of sum of all detector frames",
        title: "FFT of Sum all frames",
    },
    [AnalysisTypes.RADIAL_FOURIER]: {
        desc: "Radial Fourier analysis over the selected ring",
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