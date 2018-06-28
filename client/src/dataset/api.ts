import { DatasetState, OpenDatasetRequest, OpenDatasetResponse } from "../messages";

export function openDataset(id: string, dataset: OpenDatasetRequest): Promise<OpenDatasetResponse> {
    return fetch(`http://localhost:9000/datasets/${id}/`, {
        body: JSON.stringify(dataset),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}

export function getPreviewURL(dataset: DatasetState) {
    return `http://localhost:9000/datasets/${dataset.id}/preview/`
}