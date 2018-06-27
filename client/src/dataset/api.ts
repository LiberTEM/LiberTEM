import { OpenDatasetRequest, OpenDatasetResponse } from "../messages";

export function openDataset(id: string, dataset: OpenDatasetRequest): Promise<OpenDatasetResponse> {
    return fetch(`http://localhost:9000/datasets/${id}/`, {
        body: JSON.stringify(dataset),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}