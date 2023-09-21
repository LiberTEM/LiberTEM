export interface JobResultType {
    imageURL: string,
    description: { title: string, desc: string, includeInDownload: boolean },
}

export enum JobRunning {
    CREATING = 'CREATING',
    RUNNING = 'RUNNING',
    DONE = 'DONE',
}

export enum JobStatus {
    CREATING = 'CREATING',
    IN_PROGRESS = 'IN_PROGRESS',
    CANCELLED = 'CANCELLED',
    SUCCESS = 'SUCCESS',
    ERROR = 'ERROR',
}

export interface JobStateCommon {
    id: string,
    analysis: string,
    status: JobStatus,
    startTimestamp: number,
    results: JobResultType[],
}

export type JobStateStarted = JobStateCommon & {
    running: JobRunning.CREATING | JobRunning.RUNNING,
}

export type JobStateDone = JobStateCommon & {
    running: JobRunning.DONE,
    endTimestamp: number,
}

export type JobState = JobStateStarted | JobStateDone;