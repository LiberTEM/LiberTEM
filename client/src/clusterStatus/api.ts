import { getApiBasePath } from "../helpers/apiHelpers";
import { GetClusterDetails } from "../messages"

export async function getClusterDetail(): Promise<GetClusterDetails> {
    const basePath = getApiBasePath();
    const url = `${basePath}config/cluster/`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await r.json();
}