import { genericDelete, genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { DataSetOpenSchemaResponse, DeleteDatasetResponse, DetectDatasetResponse, OpenDatasetRequest, OpenDatasetResponse } from "../messages";

export async function openDataset(id: string, dataset: OpenDatasetRequest): Promise<OpenDatasetResponse> {
    return await genericPut(`datasets/${id}/`, dataset);
}

export async function deleteDataset(id: string): Promise<DeleteDatasetResponse> {
    return await genericDelete(`datasets/${id}/`);
}


export async function detectDataset(path: string): Promise<DetectDatasetResponse> {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}datasets/detect/?path=${encodeURIComponent(path)}`, {
        credentials: "same-origin",
        method: "GET",
    });
    return await r.json();
}

interface SchemaCache {
    [type: string]: DataSetOpenSchemaResponse,
}

const schemaCache: SchemaCache = {};

export async function getSchema(type: string): Promise<DataSetOpenSchemaResponse> {
    const basePath = getApiBasePath();
    const cached = schemaCache[type];
    if (cached) {
        return new Promise((resolve) => resolve(cached));
    } else {
        const r = await fetch(`${basePath}datasets/schema/?type=${encodeURIComponent(type)}`, {
            credentials: "same-origin",
            method: "GET",
        });
        const schemaResponse = await r.json();
        schemaCache[type] = schemaResponse;
        return schemaResponse;
    }
}