import { genericDelete, genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { AnalysisDetails, CompoundAnalysisDetails, CopyNotebookResponse, CreateAnalysisResponse, CreateCompoundAnalysisResponse, CreateOrUpdateAnalysisRequest, CreateOrUpdateCompoundAnalysisRequest, RemoveAnalysisResponse, RemoveCompoundAnalysisResponse } from "../messages";

export async function createOrUpdateAnalysis(
    compoundAnalysisId: string, analysisId: string,
    dataset: string, details: AnalysisDetails
): Promise<CreateAnalysisResponse> {
    const payload: CreateOrUpdateAnalysisRequest = {
        dataset,
        details,
    };
    return await genericPut(`compoundAnalyses/${compoundAnalysisId}/analyses/${analysisId}/`, payload);
}

export async function removeAnalysis(compoundAnalysisId: string, analysisId: string): Promise<RemoveAnalysisResponse> {
    return await genericDelete(`compoundAnalyses/${compoundAnalysisId}/analyses/${analysisId}/`)
}

export async function createOrUpdateCompoundAnalysis(compoundAnalysisId: string, dataset: string, details: CompoundAnalysisDetails): Promise<CreateCompoundAnalysisResponse> {
    const payload: CreateOrUpdateCompoundAnalysisRequest = {
        dataset,
        details,
    };
    return await genericPut(`compoundAnalyses/${compoundAnalysisId}/`, payload);
}

export async function removeCompoundAnalysis(compoundAnalysisId: string): Promise<RemoveCompoundAnalysisResponse> {
    return await genericDelete(`compoundAnalyses/${compoundAnalysisId}/`)
}

export async function getNotebook(compoundAnalysisId: string): Promise<CopyNotebookResponse>{
    const basePath = getApiBasePath();
    const url = `${basePath}compoundAnalyses/${compoundAnalysisId}/copy/notebook/`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await r.json();
}
