import { genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { ClusterDetailsResponse, ConnectRequest, ConnectRequestParams, ConnectResponse } from "../messages";

export async function connectToCluster(params: ConnectRequestParams): Promise<ConnectResponse> {
    const payload: ConnectRequest = {
        connection: params
    }
    return await genericPut("config/connection/", payload);
}

export async function checkClusterConnection(): Promise<ConnectResponse> {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}config/connection/`, {
        method: 'GET',
    });
    return await r.json();
}

export async function getClusterDetail(): Promise<ClusterDetailsResponse> {
    const basePath = getApiBasePath();
    const url = `${basePath}config/cluster/`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await r.json();
}