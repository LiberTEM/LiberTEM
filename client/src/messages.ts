
/* 
 * Common
 */

export interface FollowupPart {
    numMessages: number,
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
    isLocal: boolean,
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
    dsPath: string,
    tileshape: number[],
} & DatasetParamsCommon

export type DatasetFormParams = DatasetParamsHDF5 | DatasetParamsHDFS

export interface DatasetCreateParams {
    id: string,
    params: DatasetFormParams,
}

export type Dataset = DatasetCreateParams & {
    params: {
        shape: number[],
    }
}

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

export type MsgPartDataset = Dataset

// type alias if we want to add client-side state to datasets
export type DatasetState = Dataset

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

// tslint:disable-next-line:no-empty-interface
export interface CenterOfMassParams { }

export type CreateMaskJobRequest = MaskDefRing | MaskDefDisk

export interface StartJobRequest {
    job: {
        dataset: string,
        masks: CreateMaskJobRequest[],
    }
}

export interface StartJobResponse {
    status: "ok",
    job: string,
    details: MsgPartJob,
}