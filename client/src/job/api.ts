import { genericDelete, genericPut } from "../helpers/apiHelpers";
import { CancelJobResponse, StartJobRequest, StartJobResponse } from "../messages";

export async function startJob(jobId: string, analysis: string): Promise<StartJobResponse> {
    const payload: StartJobRequest = {
        job: {
            analysis,
        }
    }
    return await genericPut(`jobs/${jobId}/`, payload);
}

export async function cancelJob(jobId: string): Promise<CancelJobResponse> {
    return await genericDelete(`jobs/${jobId}/`);
}


