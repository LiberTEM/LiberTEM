import { genericDelete, genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { DeleteDatasetResponse, DetectDatasetResponse, OpenDatasetRequest, OpenDatasetResponse } from "../messages";

export const openDataset = async (id: string, dataset: OpenDatasetRequest): Promise<OpenDatasetResponse> => await genericPut<OpenDatasetResponse, OpenDatasetRequest>(`datasets/${id}/`, dataset)

export const deleteDataset = async (id: string): Promise<DeleteDatasetResponse> => await genericDelete<DeleteDatasetResponse>(`datasets/${id}/`)


export const detectDataset = async (path: string): Promise<DetectDatasetResponse> => {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}datasets/detect/?path=${encodeURIComponent(path)}`, {
        credentials: "same-origin",
        method: "GET",
    });
    return await (r.json() as Promise<DetectDatasetResponse>);
}
