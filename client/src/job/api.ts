import { genericDelete, genericPut } from "../helpers/apiHelpers";
import { CancelJobResponse, StartJobRequest, StartJobResponse } from "../messages";

export const startJob = async (jobId: string, analysis: string): Promise<StartJobResponse> => {
    const payload: StartJobRequest = {
        job: {
            analysis,
        }
    }
    return await genericPut<StartJobResponse, StartJobRequest>(`jobs/${jobId}/`, payload);
}

export const cancelJob = async (jobId: string): Promise<CancelJobResponse> => (
    await genericDelete<CancelJobResponse>(`jobs/${jobId}/`)
)
