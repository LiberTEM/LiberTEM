import { DeleteDatasetResponse, OpenDatasetRequest, OpenDatasetResponse } from "../messages";

export function openDataset(id: string, dataset: OpenDatasetRequest): Promise<OpenDatasetResponse> {
    return fetch(`/api/datasets/${id}/`, {
        body: JSON.stringify(dataset),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}

export function deleteDataset(id: string): Promise<DeleteDatasetResponse> {
    return fetch(`/api/datasets/${id}/`, {
        credentials: "same-origin",
        method: "DELETE",
    }).then(r => r.json());
}