import * as uuid from 'uuid/v4';

export interface OpenDatasetRequest {
    name: string,
    type: string,
    path: string,
    tileshape: number[],
}

export interface OpenDatasetResponse {
    status: "ok",
    dataset: string,  // TODO: uuid type?
}

export function openDataset(dataset: OpenDatasetRequest): Promise<OpenDatasetResponse> {
    const datasetId = uuid();
    const payload = {
        dataset,
    };
    return fetch(`http://localhost:9000/datasets/${datasetId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}