import { genericDelete } from "../helpers/apiHelpers";
import { ShutdownResponse } from '../messages'

export const doShutdown = async (): Promise<ShutdownResponse> => (
    await genericDelete<ShutdownResponse>(`shutdown/`)
);