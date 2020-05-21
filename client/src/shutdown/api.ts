import { genericDelete } from "../helpers/apiHelpers";
import { ShutdownResponse } from '../messages'

export async function doShutdown(): Promise<ShutdownResponse> {
    return await genericDelete(`shutdown/`)
}
