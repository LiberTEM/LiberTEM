
export interface FollowupPart {
    numMessages: number,
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

export interface OpenDatasetResponse {
    status: "ok",
    dataset: string,  // TODO: uuid type?
    details: Dataset,
}

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