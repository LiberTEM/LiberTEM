import { genericDelete, genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { DataSetOpenSchemaResponse, DeleteDatasetResponse, DetectDatasetResponse, OpenDatasetRequest, OpenDatasetResponse } from "../messages";

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

interface SchemaCache {
    [type: string]: DataSetOpenSchemaResponse,
}

const schemaCache: SchemaCache = {};

export const getSchema = async (type: string): Promise<DataSetOpenSchemaResponse> => {
    const basePath = getApiBasePath();
    const cached = schemaCache[type];
    if (cached) {
        return new Promise((resolve) => resolve(cached));
    } else {
        const r = await fetch(`${basePath}datasets/schema/?type=${encodeURIComponent(type)}`, {
            credentials: "same-origin",
            method: "GET",
        });
        const schemaResponse = await (r.json() as Promise<DataSetOpenSchemaResponse>);
        schemaCache[type] = schemaResponse;
        return schemaResponse;
    }
}