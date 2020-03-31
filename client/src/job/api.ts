import { getApiBasePath } from "../helpers/apiHelpers";
import { CancelJobResponse, StartJobRequest, StartJobResponse } from "../messages";

export function startJob(jobId: string, analysis: string): Promise<StartJobResponse> {
    const payload: StartJobRequest = {
        job: {
            analysis,
        }
    }
    const basePath = getApiBasePath();
    return fetch(`${basePath}jobs/${jobId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}

export function cancelJob(jobId: string): Promise<CancelJobResponse> {
    const basePath = getApiBasePath();
    return fetch(`${basePath}jobs/${jobId}/`, {
        method: "DELETE",
        credentials: "same-origin",
    }).then(r => r.json());
}


