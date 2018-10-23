
export interface JobResultType {
    imageURL: string,
    description: { title: string, desc: string },
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

export interface JobState {
    id: string,
    dataset: string,
    running: JobRunning,
    status: JobStatus,
    results: JobResultType[],
    startTimestamp: number,
    endTimestamp: number,
}