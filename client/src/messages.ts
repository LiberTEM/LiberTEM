
export interface FollowupPart {
    numMessages: number,
}

/*
 * Dataset
 */
export interface DatasetCreateParams {
    id: string,
    name: string,
    path: string,
    tileshape: number[],
    type: string,
}

export type Dataset = DatasetCreateParams & {
    shape: number[],
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