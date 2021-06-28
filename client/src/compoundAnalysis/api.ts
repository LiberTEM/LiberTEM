import { genericDelete, genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { AnalysisDetails, CompoundAnalysisDetails, CopyNotebookResponse, CreateAnalysisResponse, CreateCompoundAnalysisResponse, CreateOrUpdateAnalysisRequest, CreateOrUpdateCompoundAnalysisRequest, RemoveAnalysisResponse, RemoveCompoundAnalysisResponse } from "../messages";

export const createOrUpdateAnalysis = async (
    compoundAnalysisId: string,
    analysisId: string, dataset: string, details: AnalysisDetails
): Promise<CreateAnalysisResponse> => {
    const payload: CreateOrUpdateAnalysisRequest = {
        dataset,
        details,
    };
    return await genericPut(`compoundAnalyses/${compoundAnalysisId}/analyses/${analysisId}/`, payload);
}

export const removeAnalysis = async (compoundAnalysisId: string, analysisId: string): Promise<RemoveAnalysisResponse> => (
    await genericDelete(`compoundAnalyses/${compoundAnalysisId}/analyses/${analysisId}/`)
);

export const createOrUpdateCompoundAnalysis = async (
    compoundAnalysisId: string,
    dataset: string,
    details: CompoundAnalysisDetails,
): Promise<CreateCompoundAnalysisResponse> => {
    const payload: CreateOrUpdateCompoundAnalysisRequest = {
        dataset,
        details,
    };
    return await genericPut(`compoundAnalyses/${compoundAnalysisId}/`, payload);
}

export const removeCompoundAnalysis = async (compoundAnalysisId: string): Promise<RemoveCompoundAnalysisResponse> => (
    await genericDelete(`compoundAnalyses/${compoundAnalysisId}/`)
);

export const getNotebook = async (compoundAnalysisId: string): Promise<CopyNotebookResponse> => {
    const basePath = getApiBasePath();
    const url = `${basePath}compoundAnalyses/${compoundAnalysisId}/copy/notebook/`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await (r.json() as Promise<CopyNotebookResponse>);
}
