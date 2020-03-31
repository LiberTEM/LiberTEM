import { getApiBasePath } from "../helpers/apiHelpers";
import { GetConfigResponse } from "../messages";

export function getConfig(): Promise<GetConfigResponse> {
    const basePath = getApiBasePath();
    return fetch(`${basePath}config/`, {
        method: "GET",
        credentials: "same-origin",
    }).then(r => r.json());
}