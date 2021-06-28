import { getApiBasePath } from "../helpers/apiHelpers";
import { ClusterDetailsResponse } from "../messages";

export const getClusterDetail = async (): Promise<ClusterDetailsResponse> => {
    const basePath = getApiBasePath();
    const url = `${basePath}config/cluster/`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await (r.json() as Promise<ClusterDetailsResponse>);
}