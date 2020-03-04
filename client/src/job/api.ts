import { CancelJobResponse, StartJobRequest, StartJobResponse } from "../messages";
import { AnalysisDetails, CreateAnalysisResponse, CreateOrUpdateAnalysisRequest, RemoveAnalysisResponse } from '../messages';

export function startJob(jobId: string, analysis: string): Promise<StartJobResponse> {
    const payload: StartJobRequest = {
        job: {
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


export async function createOrUpdateAnalysis(
    analysisId: string, dataset: string, details: AnalysisDetails
): Promise<CreateAnalysisResponse> {
    const payload: CreateOrUpdateAnalysisRequest = {
        dataset,
        analysis: details,
    }

    const r = await fetch(`/api/analyses/${analysisId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    });
    return await r.json();
}

export async function removeAnalysis(analysisId: string): Promise<RemoveAnalysisResponse> {
    const r = await fetch(`/api/analyses/${analysisId}/`, {
        credentials: "same-origin",
        method: "DELETE",
    });
    return await r.json();
}