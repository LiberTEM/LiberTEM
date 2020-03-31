import { getApiBasePath } from "../helpers/apiHelpers";
import { ConnectRequest, ConnectRequestParams, ConnectResponse } from "../messages";

export function connectToCluster(params: ConnectRequestParams): Promise<ConnectResponse> {
    const payload: ConnectRequest = {
        connection: params
    }
    const basePath = getApiBasePath();
    return fetch(`${basePath}config/connection/`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    }).then(r => r.json());
}

export function checkClusterConnection(): Promise<ConnectResponse> {
    const basePath = getApiBasePath();
    return fetch(`${basePath}config/connection/`, {
        method: 'GET',
    }).then(r => r.json());
}