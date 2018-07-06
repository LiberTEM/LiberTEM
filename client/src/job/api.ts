import { AnalysisDetails, CancelJobResponse, StartJobRequest, StartJobResponse } from "../messages";

export function startJob(jobId: string, datasetId: string, analysis: AnalysisDetails): Promise<StartJobResponse> {
    const payload: StartJobRequest = {
        job: {
            dataset: datasetId,
            analysis,
        }
    }
    return fetch(`/api/jobs/${jobId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}

export function cancelJob(jobId: string): Promise<CancelJobResponse> {
    return fetch(`/api/jobs/${jobId}/`, {
        method: "DELETE",
        credentials: "same-origin",
    }).then(r => r.json());
}