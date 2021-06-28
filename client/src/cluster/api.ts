import { genericPut, getApiBasePath } from "../helpers/apiHelpers";
import { ConnectRequest, ConnectRequestParams, ConnectResponse } from "../messages";

export const connectToCluster = async (params: ConnectRequestParams): Promise<ConnectResponse> => {
    const payload: ConnectRequest = {
        connection: params
    }
    return await genericPut("config/connection/", payload);
}

export const checkClusterConnection = async (): Promise<ConnectResponse> => {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}config/connection/`, {
        method: 'GET',
    });
    return await (r.json() as Promise<ConnectResponse>);
}
