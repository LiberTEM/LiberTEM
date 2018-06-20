import * as uuid from 'uuid/v4';

export interface MaskDefRing {
    shape: "ring",
    cx: number,
    cy: number,
    ri: number,
    ro: number
}

export interface MaskDefDisk {
    shape: "disk",
    cx: number,
    cy: number,
    r: number,
}

export type CreateMaskJobRequest = MaskDefRing | MaskDefDisk

export function startJob(datasetId: string, masks: CreateMaskJobRequest[]) {
    const jobId = uuid();
    const payload = {
        job: {
            dataset: datasetId,
            masks,
        }
    }
    return fetch(`http://localhost:9000/jobs/${jobId}/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}