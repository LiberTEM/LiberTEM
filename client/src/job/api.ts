import { CreateMaskJobRequest, StartJobRequest, StartJobResponse } from "../messages";

export function startJob(jobId: string, datasetId: string, masks: CreateMaskJobRequest[]): Promise<StartJobResponse> {
    const payload: StartJobRequest = {
        job: {
            dataset: datasetId,
            masks,
        }
    }
    return fetch(`/api/jobs/${jobId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}