
/* 
 * Common
 */

export interface FollowupPart {
    numMessages: number,
    descriptions: Array<{ title: string, desc: string }>,
}

export interface MsgPartConfig {
    version: string,
    revision: string,
    localCores: number,
    cwd: string,
    separator: string,
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
    connection: {},
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
}

export interface DatasetParamsCommon {
    name: string,
}

export type DatasetParamsHDF5 = {
    type: DatasetTypes.HDF5,
    path: string,
    ds_path: string,
    tileshape: number[],
} & DatasetParamsCommon

export type DatasetParamsRaw = {
    type: DatasetTypes.RAW,
    path: string,
    dtype: string,
    detector_size: number[],
    enable_direct: boolean,
    scan_size: number[],
    tileshape: number[],
} & DatasetParamsCommon

export type DatasetParamsMIB = {
    type: DatasetTypes.MIB,
    path: string,
    scan_size: number[],
    tileshape: number[],
} & DatasetParamsCommon

export type DatasetParamsBLO = {
    type: DatasetTypes.BLO,
    path: string,
    tileshape: number[],
} & DatasetParamsCommon

export type DatasetParamsK2IS = {
    type: DatasetTypes.K2IS,
    path: string,
} & DatasetParamsCommon

export type DatasetParamsSER = {
    type: DatasetTypes.SER,
    path: string,
} & DatasetParamsCommon

export type DatasetParamsFRMS6 = {
    type: DatasetTypes.FRMS6,
    path: string,
} & DatasetParamsCommon

export type DatasetParamsEMPAD = {
    type: DatasetTypes.EMPAD,
    path: string,
    scan_size: number[],
} & DatasetParamsCommon

export type DatasetFormParams = DatasetParamsHDF5 | DatasetParamsRaw | DatasetParamsMIB | DatasetParamsBLO | DatasetParamsK2IS | DatasetParamsSER | DatasetParamsFRMS6 | DatasetParamsEMPAD

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

export type Dataset = DatasetOpening | DatasetOpen | DatasetDeleting;

export interface OpenDatasetRequest {
    dataset: DatasetCreateParams
}

export interface OpenDatasetResponseOk {
    status: "ok",
    dataset: string,  // TODO: uuid type?
    details: Dataset,
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
}

export interface DetectDatasetErrorResponse {
    status: "error",
    path: string,
    msg: string,
}

export type DetectDatasetResponse = DetectDatasetSuccessResponse | DetectDatasetErrorResponse;

export interface DataSetOpenSchemaSuccessResponse {
    status: "ok",
    ds_type: string,
    schema: object,
}

export interface DataSetOpenSchemaErrorResponse {
    status: "error",
    ds_type: string,
    msg: string,
}

export type DataSetOpenSchemaResponse = DataSetOpenSchemaSuccessResponse | DataSetOpenSchemaErrorResponse;

export type MsgPartInitialDataset = DatasetOpen

// type alias to add client-side state to datasets
export type DatasetState = Dataset & {}

/*
 * Job
 */
export interface MsgPartJob {
    id: string,
    dataset: string,
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


export interface FrameParams {
    roi: {
        shape: "rect",
        x: number,
        y: number,
        width: number,
        height: number,
    } | {
        shape: "disk",
        cx: number,
        cy: number,
        r: number,
    } |
    {}
}

export interface ClustParams {
    roi: {
        shape: "rect",
        x: number,
        y: number,
        width: number,
        height: number,
    } | {}
    cx: number,
    cy: number,
    ri: number,
    ro: number,
    delta: number,
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
    type: AnalysisTypes.APPLY_RING_MASK,
    parameters: MaskDefRing,

}

export interface FFTDetails {
    type: AnalysisTypes.APPLY_FFT_MASK,
    parameters: FFTParams,
}

export interface FEMDetails {
    type: AnalysisTypes.FEM,
    parameters: MaskDefRing,
}

export interface DiskMaskDetails {
    type: AnalysisTypes.APPLY_DISK_MASK,
    parameters: MaskDefDisk,
}

export interface PointDefDetails {
    type: AnalysisTypes.APPLY_POINT_SELECTOR,
    parameters: PointDef,
}

export interface CenterOfMassDetails {
    type: AnalysisTypes.CENTER_OF_MASS,
    parameters: CenterOfMassParams,
}

export interface SumFramesDetails {
    type: AnalysisTypes.SUM_FRAMES,
    parameters: FrameParams
}

export interface SDFramesDetails {
    type: AnalysisTypes.SD_FRAMES,
    parameters: FrameParams
}

export interface SumSigDetails {
    type: AnalysisTypes.SUM_SIG,
    parameters: {}
}

export interface FFTSumFramesDetails {
    type: AnalysisTypes.FFTSUM_FRAMES,
    parameters: FFTSumFramesParams,
}

export interface PickFrameDetails {
    type: AnalysisTypes.PICK_FRAME,
    parameters: PickFrameParams,
}

export interface PickFFTFrameDetails {
    type: AnalysisTypes.PICK_FFT_FRAME,
    parameters: PickFFTFrameParams,
}

export interface RadialFourierDetails {
    type: AnalysisTypes.RADIAL_FOURIER,
    parameters: RadialFourierParams,
}

export interface RadialFourierDetails {
    type: AnalysisTypes.RADIAL_FOURIER,
    parameters: RadialFourierParams,
}

export interface ClustDetails {
    type: AnalysisTypes.CLUST,
    parameters: ClustParams,
}

export type AnalysisParameters = MaskDefRing | MaskDefDisk | CenterOfMassParams | PointDef | PickFrameParams | RadialFourierParams | FFTParams | PickFFTFrameParams | FFTSumFramesParams | ClustParams;
export type AnalysisDetails = RingMaskDetails | DiskMaskDetails | CenterOfMassDetails | PointDefDetails | SumFramesDetails | SDFramesDetails | PickFrameDetails | RadialFourierDetails | FEMDetails | FFTDetails | FFTSumFramesDetails | PickFFTFrameDetails | SumSigDetails | ClustDetails;

export interface StartJobRequest {
    job: {
        dataset: string,
        analysis: AnalysisDetails,
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
