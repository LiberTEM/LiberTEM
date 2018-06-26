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

// tslint:disable-next-line:no-empty-interface
export interface CenterOfMassParams { }

export type CreateMaskJobRequest = MaskDefRing | MaskDefDisk

export interface StartJobResponse {
    status: "ok",
    job: string,
}

export function startJob(jobId: string, datasetId: string, masks: CreateMaskJobRequest[]): Promise<StartJobResponse> {
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