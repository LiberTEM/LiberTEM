import { JobList } from "./analysis/types"

/*
 * Common
 */

export interface FollowupPart {
    numMessages: number,
    descriptions: Array<{ title: string, desc: string, includeInDownload: boolean }>,
}

export interface ResultFileFormat {
    identifier: string,
    description: string,
}

export interface ProgressDetails {
    event: "start" | "update" | "end",
    numFrames: number,
    numFramesComplete: number,
}

export type JsonSchema = Record<string, unknown>;

export interface DatasetTypeInfo {
    schema: JsonSchema,
    default_io_backend: IOBackendId | null,
    supported_io_backends: IOBackendId[],
}

export interface MsgPartConfig {
    version: string,
    revision: string,
    localCores: number,
    devices: {
        cpus: number[],
        cudas: number[],
        has_cupy: boolean,
    }
    cwd: string,
    separator: string,
    resultFileFormats: {
        [id: string]: ResultFileFormat,
    },
    datasetTypes: {
        [id: string]: DatasetTypeInfo,
    },
}

export interface GetConfigResponse {
    status: "ok",
    config: MsgPartConfig,
}

/*
 * Connection
 */

export enum ClusterTypes {
    LOCAL = "LOCAL",
    TCP = "TCP",
}

export const ClusterTypeMetadata: { [s: string]: { [s: string]: string } } = {
    [ClusterTypes.LOCAL]: {
        label: "Create local cluster",
    },
    [ClusterTypes.TCP]: {
        label: "Connect to cluster",
        helpText: "can be either local or remote, connection via TCP",
    }
}

export interface ConnectRequestLocalCluster {
    type: ClusterTypes.LOCAL,
    numWorkers?: number,
    cudas: Record<number, number>
}

export interface ConnectRequestTCP {
    type: ClusterTypes.TCP,
    address: string,
}

export type ConnectRequestParams = ConnectRequestLocalCluster | ConnectRequestTCP

export interface ConnectRequest {
    connection: ConnectRequestParams
}

export type ConnectResponse = {
    status: "ok",
    connection: ConnectRequest,
} | {
    status: "disconnected",
    connection: Record<string, never>,
} | {
    status: "error",
    messageType: string,
    msg: string,
}

export interface HostDetails {
    host: string,
    cpu: number,
    cuda: number,
    service: number,
}

export interface ClusterDetailsResponse {
    status: "ok",
    messageType: string,
    details: HostDetails[],
}

/*
 * Dataset
 */

export enum DatasetTypes {
    HDF5 = "HDF5",
    RAW = "RAW",
    MIB = "MIB",
    BLO = "BLO",
    K2IS = "K2IS",
    SER = "SER",
    FRMS6 = "FRMS6",
    EMPAD = "EMPAD",
    SEQ = "SEQ",
    MRC = "MRC",
    TVIPS = "TVIPS",
    NPY = "NPY",
    RAW_CSR = "RAW_CSR",
    DM = 'DM',
}

export interface DatasetParamsCommon {
    name: string,
    nav_shape: number[],
    sig_shape: number[],
    sync_offset: number,
    io_backend?: IOBackendId,

    // deprecated, for loading old localStorage values:
    scan_size?: number[],
    detector_size?: number[],
}

export interface DatasetInfoCommon {
    image_count: number,
    native_sig_shape: number[],
}

export type DatasetParamsHDF5 = {
    type: DatasetTypes.HDF5,
    path: string,
    ds_path: string,
} & DatasetParamsCommon

export interface DatasetInfoHDF5Item {
    path: string,
    shape: number[],
    compression: null | string,
    chunks: null | number[],
    raw_nav_shape: number[],
    nav_shape: number[],
    sig_shape: number[],
    image_count: number,
}

export type DatasetInfoHDF5 = {
    type: DatasetTypes.HDF5,
    datasets?: DatasetInfoHDF5Item[],
} & DatasetInfoCommon

export type DatasetParamsRaw = {
    type: DatasetTypes.RAW,
    path: string,
    dtype: string,
} & DatasetParamsCommon

export interface DatasetInfoRAW {
    type: DatasetTypes.RAW,
}

export type DatasetParamsMIB = {
    type: DatasetTypes.MIB,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoMIB = {
    type: DatasetTypes.MIB,
} & DatasetInfoCommon

export type DatasetParamsNPY = {
    type: DatasetTypes.NPY,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoNPY = {
    type: DatasetTypes.NPY,
} & DatasetInfoCommon

export type DatasetParamsBLO = {
    type: DatasetTypes.BLO,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoBLO = {
    type: DatasetTypes.BLO,
} & DatasetInfoCommon

export type DatasetParamsK2IS = {
    type: DatasetTypes.K2IS,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoK2IS = {
    type: DatasetTypes.K2IS,
} & DatasetInfoCommon

export type DatasetParamsSER = {
    type: DatasetTypes.SER,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoSER = {
    type: DatasetTypes.SER,
} & DatasetInfoCommon

export type DatasetParamsFRMS6 = {
    type: DatasetTypes.FRMS6,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoFRMS6 = {
    type: DatasetTypes.FRMS6,
} & DatasetInfoCommon

export type DatasetParamsEMPAD = {
    type: DatasetTypes.EMPAD,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoEMPAD = {
    type: DatasetTypes.EMPAD,
} & DatasetInfoCommon

export type DatasetParamsSEQ = {
    type: DatasetTypes.SEQ,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoSEQ = {
    type: DatasetTypes.SEQ,
} & DatasetInfoCommon

export type DatasetParamsMRC = {
    type: DatasetTypes.MRC,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoMRC = {
    type: DatasetTypes.MRC,
} & DatasetInfoCommon


export type DatasetParamsTVIPS = {
    type: DatasetTypes.TVIPS,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoTVIPS = {
    type: DatasetTypes.TVIPS,
} & DatasetInfoCommon

export type DatasetParamsRawCSR = {
    type: DatasetTypes.RAW_CSR,
    path: string,
} & DatasetParamsCommon

export type DatasetInfoRawCSR = {
    type: DatasetTypes.RAW_CSR,
    path: string,
} & DatasetInfoCommon

export type DatasetParamsDM = {
    type: DatasetTypes.DM,
    path: string,
    force_c_order: boolean,
} & DatasetParamsCommon

export type DatasetInfoDM = {
    type: DatasetTypes.DM,
} & DatasetInfoCommon


export type DatasetFormParams = DatasetParamsHDF5 | DatasetParamsRaw | DatasetParamsMIB | DatasetParamsBLO | DatasetParamsK2IS | DatasetParamsSER | DatasetParamsFRMS6 | DatasetParamsEMPAD | DatasetParamsSEQ | DatasetParamsMRC | DatasetParamsTVIPS | DatasetParamsNPY | DatasetParamsRawCSR | DatasetParamsDM
export type DatasetFormInfo = DatasetInfoHDF5 | DatasetInfoRAW | DatasetInfoMIB | DatasetInfoBLO | DatasetInfoK2IS | DatasetInfoSER | DatasetInfoFRMS6 | DatasetInfoEMPAD | DatasetInfoSEQ | DatasetInfoMRC | DatasetInfoTVIPS | DatasetInfoNPY | DatasetInfoRawCSR | DatasetInfoDM

export interface DatasetCreateParams {
    id: string,
    params: DatasetFormParams,
}

export enum DatasetStatus {
    OPEN = "OPEN",
    OPENING = "OPENING",
    DELETING = "DELETING",
}

export interface DiagElemMsg {
    name: string,
    value: string | DiagElemMsg[],
}

interface DatasetCommon {
    id: string,
    params: DatasetFormParams,
}

export type DatasetOpening = DatasetCommon & {
    status: DatasetStatus.OPENING,
}

export type DatasetDeleting = DatasetCommon & {
    status: DatasetStatus.DELETING,
}

export type DatasetOpen = DatasetCommon & {
    status: DatasetStatus.OPEN,
    params: {
        shape: number[],
    }
    diagnostics: DiagElemMsg[],
}

export type CreateDatasetMessage = Omit<DatasetOpen, "status">;

export type Dataset = DatasetOpening | DatasetOpen | DatasetDeleting;

export type IOBackendId = "direct" | "buffered" | "mmap";

export const IOBackendMetadata: { [s in IOBackendId]: { label: string } } = {
    "mmap": {
        label: "MMAP I/O backend (default on Linux)",
    },
    "buffered": {
        label: "Buffered I/O backend, useful for slow HDD storage (default on Windows)",
    },
    "direct": {
        label: "Direct I/O backend, useful for very large data on fast storage",
    },
};

export interface OpenDatasetRequest {
    dataset: DatasetCreateParams,
}

export interface OpenDatasetResponseOk {
    status: "ok",
    dataset: string,  // TODO: uuid type?
    details: DatasetOpen,
}

export interface OpenDatasetResponseError {
    status: "error",
    dataset: string,
    msg: string,
}

export type OpenDatasetResponse = OpenDatasetResponseOk | OpenDatasetResponseError

export interface DeleteDatasetResponse {
    status: "ok",
    dataset: string,
}

export interface DetectDatasetSuccessResponse {
    status: "ok",
    datasetParams: DatasetFormParams,
    datasetInfo: DatasetFormInfo,
}

export interface DetectDatasetErrorResponse {
    status: "error",
    path: string,
    msg: string,
}

export type DetectDatasetResponse = DetectDatasetSuccessResponse | DetectDatasetErrorResponse;

export type MsgPartInitialDataset = DatasetOpen

// type alias to add client-side state to datasets (currently empty)
export type DatasetState = Dataset

export const ShapeLengths = {
    NAV_SHAPE_MIN_LENGTH: 2,
    NAV_SHAPE_MAX_LENGTH: 2,
    SIG_SHAPE_MIN_LENGTH: 2,
    SIG_SHAPE_MAX_LENGTH: 2
};

/*
 * Job
 */
export interface MsgPartJob {
    id: string,
    analysis: string,
}

export interface MaskDefRing {
    shape: "ring",
    cx: number,
    cy: number,
    ri: number,
    ro: number
}


export interface MaskDefDisk {
    shape: "disk",
    cx: number,
    cy: number,
    r: number,
}

export interface PointDef {
    shape: "point",
    cx: number,
    cy: number,
}

// TODO: shape doesn't really make sense here, needs to be restructured
export interface CenterOfMassParams {
    shape: "com",
    cx: number,
    cy: number,
    r: number,
    ri?: number,
    flip_y: boolean,
    scan_rotation: number,
}

export interface PickFrameParams {
    x: number,
    y: number,
}

export interface FFTSumFramesParams {
    real_rad: number | null,
    real_centerx: number | null,
    real_centery: number | null,
}

export interface PickFFTFrameParams {
    x: number,
    y: number,
    real_rad: number | null,
    real_centerx: number | null,
    real_centery: number | null,
}
export interface RadialFourierParams {
    shape: "radial_fourier",
    cx: number,
    cy: number,
    ri: number,
    ro: number,
    n_bins: number,
    max_order: number
}

export interface FFTParams {
    rad_in: number,
    rad_out: number,
    real_rad: number | null,
    real_centerx: number | null,
    real_centery: number | null,
}

export interface RectRoiParams {
    shape: "rect",
    x: number,
    y: number,
    width: number,
    height: number,
}

export interface DiskRoiParams {
    shape: "disk",
    cx: number,
    cy: number,
    r: number,
}

export interface FrameParams {
    roi: RectRoiParams | DiskRoiParams | Record<string, never>
}

export interface ClustParams {
    roi: RectRoiParams | Record<string, never>,
    cx: number,
    cy: number,
    ri: number,
    ro: number,
    n_peaks: number,
    n_clust: number,
    min_dist: number,
}

export enum AnalysisTypes {
    APPLY_RING_MASK = "APPLY_RING_MASK",
    APPLY_DISK_MASK = "APPLY_DISK_MASK",
    APPLY_POINT_SELECTOR = "APPLY_POINT_SELECTOR",
    CENTER_OF_MASS = "CENTER_OF_MASS",
    SUM_FRAMES = "SUM_FRAMES",
    SD_FRAMES = "SD_FRAMES",
    PICK_FRAME = "PICK_FRAME",
    PICK_FFT_FRAME = "PICK_FFT_FRAME",
    APPLY_FFT_MASK = "APPLY_FFT_MASK",
    FFTSUM_FRAMES = "FFTSUM_FRAMES",
    RADIAL_FOURIER = "RADIAL_FOURIER",
    FEM = "FEM",
    CLUST = "CLUST",
    SUM_SIG = "SUM_SIG",
}

export interface RingMaskDetails {
    analysisType: AnalysisTypes.APPLY_RING_MASK,
    parameters: MaskDefRing,
}

export interface FFTDetails {
    analysisType: AnalysisTypes.APPLY_FFT_MASK,
    parameters: FFTParams,
}

export interface FEMDetails {
    analysisType: AnalysisTypes.FEM,
    parameters: MaskDefRing,
}

export interface DiskMaskDetails {
    analysisType: AnalysisTypes.APPLY_DISK_MASK,
    parameters: MaskDefDisk,
}

export interface PointDefDetails {
    analysisType: AnalysisTypes.APPLY_POINT_SELECTOR,
    parameters: PointDef,
}

export interface CenterOfMassDetails {
    analysisType: AnalysisTypes.CENTER_OF_MASS,
    parameters: CenterOfMassParams,
}

export interface SumFramesDetails {
    analysisType: AnalysisTypes.SUM_FRAMES,
    parameters: FrameParams
}

export interface SDFramesDetails {
    analysisType: AnalysisTypes.SD_FRAMES,
    parameters: FrameParams
}

export interface SumSigDetails {
    analysisType: AnalysisTypes.SUM_SIG,
    parameters: Record<string, never>
}

export interface FFTSumFramesDetails {
    analysisType: AnalysisTypes.FFTSUM_FRAMES,
    parameters: FFTSumFramesParams,
}

export interface PickFrameDetails {
    analysisType: AnalysisTypes.PICK_FRAME,
    parameters: PickFrameParams,
}

export interface PickFFTFrameDetails {
    analysisType: AnalysisTypes.PICK_FFT_FRAME,
    parameters: PickFFTFrameParams,
}

export interface RadialFourierDetails {
    analysisType: AnalysisTypes.RADIAL_FOURIER,
    parameters: RadialFourierParams,
}

export interface ClustDetails {
    analysisType: AnalysisTypes.CLUST,
    parameters: ClustParams,
}

export type AnalysisParameters = MaskDefRing | MaskDefDisk | CenterOfMassParams | PointDef | PickFrameParams | RadialFourierParams | FFTParams | PickFFTFrameParams | FFTSumFramesParams | ClustParams;
export type AnalysisDetails = RingMaskDetails | DiskMaskDetails | CenterOfMassDetails | PointDefDetails | SumFramesDetails | SDFramesDetails | PickFrameDetails | RadialFourierDetails | FEMDetails | FFTDetails | FFTSumFramesDetails | PickFFTFrameDetails | SumSigDetails | ClustDetails;

export interface MsgPartAnalysis {
    analysis: string,
    dataset: string,
    details: AnalysisDetails,
    jobs: JobList,
}

export type CreateOrUpdateAnalysisRequest = Omit<MsgPartAnalysis, "analysis" | "jobs">;

export type CreateAnalysisResponse = {
    status: "ok",
    messageType: "ANALYSIS_CREATED",
} & MsgPartAnalysis

export type UpdateAnalysisResponse = {
    status: "ok",
    messageType: "ANALYSIS_UPDATED",
} & MsgPartAnalysis

export type RemoveAnalysisResponse = {
    status: "ok"
    messageType: "ANALYSIS_REMOVED",
    analysis: string,
} | {
    status: "error",
    messageType: "ANALYSIS_REMOVAL_FAILED",
    msg: string,
    analysis: string,
}

export interface CompoundAnalysisDetails {
    mainType: AnalysisTypes,
    analyses: string[],
}

export interface MsgPartCompoundAnalysis {
    compoundAnalysis: string,
    dataset: string,
    details: CompoundAnalysisDetails,
}

export interface CreateOrUpdateCompoundAnalysisRequest {
    dataset: string,
    details: CompoundAnalysisDetails,
}

export type CreateCompoundAnalysisResponse = {
    status: "ok",
    messageType: "COMPOUND_ANALYSIS_CREATED",
} & MsgPartCompoundAnalysis;

export type RemoveCompoundAnalysisResponse = {
    status: "ok"
    messageType: "COMPOUND_ANALYSIS_REMOVED",
    analysis: string,
} | {
    status: "error",
    messageType: "COMPOUND_ANALYSIS_REMOVAL_FAILED",
    msg: string,
    analysis: string,
}

export interface StartJobRequest {
    job: {
        analysis: string,
    }
}

export interface StartJobResponse {
    status: "ok",
    job: string,
    details: MsgPartJob,
}

export interface CancelJobResponse {
    status: "ok",
    job: string,
}

/*
 * fs browser
 */

// some named place, i.e. "Documents", "Home", ...
export interface FSPlace {
    title: string,
    path: string,
    key: string,
}

export interface DirectoryListingDetails {
    name: string,
    size: number,
    ctime: number,
    mtime: number,
    owner: string,
}

export interface DirectoryListingResponseOK {
    status: "ok",
    path: string,
    files: DirectoryListingDetails[],
    dirs: DirectoryListingDetails[],
    drives: string[],
    places: FSPlace[],
}

export interface DirectoryListingResponseError {
    status: "error",
    path: string,
    code: string,
    msg: string,
    alternative?: string,
}

export type DirectoryListingResponse = DirectoryListingResponseOK | DirectoryListingResponseError;


export interface StatResponseOK {
    status: "ok",
    path: string,
    dirname: string,
    basename: string,
    stat: {
        size: number,
        ctime: number,
        mtime: number,
        owner: string,
        isdir: boolean,
        isreg: boolean,
    },
}

export interface StatResponseError {
    status: "error",
    path: string,
    code: string,
    msg: string,
    alternative?: string,
}

export type StatResponse = StatResponseError | StatResponseOK;


export interface ShutdownResponse {
    status: "ok",
    messageType: "SERVER_SHUTDOWN",
}

export interface  CopyAnalysis{
    analysis: string,
    plot: string[],
}

export interface CopyNotebookResponse {
    dependency: string,
    initial_setup: string,
    ctx: string,
    dataset: string,
    analysis: CopyAnalysis[],
}
