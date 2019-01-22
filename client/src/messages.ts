
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
    HDFS = "HDFS",
    RAW = "RAW",
    MIB = "MIB",
    BLO = "BLO",
    K2IS = "K2IS",
    SER = "SER",
}

export interface DatasetParamsCommon {
    name: string,
}

export type DatasetParamsHDFS = {
    type: DatasetTypes.HDFS,
    path: string,
    tileshape: number[],
} & DatasetParamsCommon;

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
    detector_size_raw: number[],
    crop_detector_to: number[],
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

export type DatasetFormParams = DatasetParamsHDF5 | DatasetParamsHDFS | DatasetParamsRaw | DatasetParamsMIB | DatasetParamsBLO | DatasetParamsK2IS | DatasetParamsSER

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

export enum AnalysisTypes {
    APPLY_RING_MASK = "APPLY_RING_MASK",
    APPLY_DISK_MASK = "APPLY_DISK_MASK",
    APPLY_POINT_SELECTOR = "APPLY_POINT_SELECTOR",
    CENTER_OF_MASS = "CENTER_OF_MASS",
    SUM_FRAMES = "SUM_FRAMES",
    PICK_FRAME = "PICK_FRAME",
}

export interface RingMaskDetails {
    type: AnalysisTypes.APPLY_RING_MASK,
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
    parameters: {},
}

export interface PickFrameDetails {
    type: AnalysisTypes.PICK_FRAME,
    parameters: PickFrameParams,
}

export type AnalysisParameters = MaskDefRing | MaskDefDisk | CenterOfMassParams | PointDef | PickFrameParams;
export type AnalysisDetails = RingMaskDetails | DiskMaskDetails | CenterOfMassDetails | PointDefDetails | SumFramesDetails | PickFrameDetails;

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
