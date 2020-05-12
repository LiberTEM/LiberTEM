import { genericDelete } from "../helpers/apiHelpers";
import { ShutdownResponse } from '../messages'

export async function handleSubmit(): Promise<ShutdownResponse> {
    return await genericDelete(`shutdown/`)
}
