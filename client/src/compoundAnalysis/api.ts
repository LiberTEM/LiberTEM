import { getApiBasePath } from "../helpers/apiHelpers";
import {
    AnalysisDetails, CompoundAnalysisDetails, CreateAnalysisResponse,
    CreateCompoundAnalysisResponse, CreateOrUpdateAnalysisRequest,
    CreateOrUpdateCompoundAnalysisRequest, RemoveAnalysisResponse,
    RemoveCompoundAnalysisResponse
} from "../messages";

export async function createOrUpdateAnalysis(
    compoundAnalysisId: string, analysisId: string,
    dataset: string, details: AnalysisDetails
): Promise<CreateAnalysisResponse> {
    const payload: CreateOrUpdateAnalysisRequest = {
        dataset,
        details,
    };
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}compoundAnalyses/${compoundAnalysisId}/analyses/${analysisId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    });
    return await r.json();
}

export async function removeAnalysis(compoundAnalysisId: string, analysisId: string): Promise<RemoveAnalysisResponse> {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}compoundAnalyses/${compoundAnalysisId}/analyses/${analysisId}/`, {
        credentials: "same-origin",
        method: "DELETE",
    });
    return await r.json();
}

export async function createOrUpdateCompoundAnalysis(compoundAnalysisId: string, dataset: string, details: CompoundAnalysisDetails): Promise<CreateCompoundAnalysisResponse> {
    const basePath = getApiBasePath();
    const payload: CreateOrUpdateCompoundAnalysisRequest = {
        dataset,
        details,
    };
    const r = await fetch(`${basePath}compoundAnalyses/${compoundAnalysisId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    });
    return await r.json();
}

export async function removeCompoundAnalysis(compoundAnalysisId: string): Promise<RemoveCompoundAnalysisResponse> {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}compoundAnalyses/${compoundAnalysisId}/`, {
        credentials: "same-origin",
        method: "DELETE",
    });
    return await r.json();
}
