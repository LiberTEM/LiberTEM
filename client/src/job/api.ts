import { AnalysisDetails, CancelJobResponse, CompoundAnalysisDetails, CreateAnalysisResponse, CreateCompoundAnalysisResponse, CreateOrUpdateAnalysisRequest, CreateOrUpdateCompoundAnalysisRequest, RemoveAnalysisResponse, RemoveCompoundAnalysisResponse, StartJobRequest, StartJobResponse } from "../messages";

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
        details,
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

export async function createOrUpdateCompoundAnalysis(
    compoundAnalysisId: string, dataset: string, details: CompoundAnalysisDetails,
): Promise<CreateCompoundAnalysisResponse> {
    const payload: CreateOrUpdateCompoundAnalysisRequest = {
        dataset,
        details,
    }

    const r = await fetch(`/api/compoundAnalyses/${compoundAnalysisId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    });
    return await r.json();
}

export async function removeCompoundAnalysis(
    compoundAnalysisId: string
): Promise<RemoveCompoundAnalysisResponse> {

    const r = await fetch(`/api/compoundAnalyses/${compoundAnalysisId}/`, {
        credentials: "same-origin",
        method: "DELETE",
    });
    return await r.json();
}