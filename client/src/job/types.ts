export interface JobResultType {
    imageURL: string,
}

export interface Job {
    id: string,
    dataset: string,
    running: "RUNNING" | "DONE",
    status: "IN_PROGRESS" | "CANCELLED" | "SUCCESS",
    results: JobResultType[],
}