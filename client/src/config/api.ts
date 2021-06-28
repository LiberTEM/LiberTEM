import { getApiBasePath } from "../helpers/apiHelpers";
import { GetConfigResponse } from "../messages";

export const getConfig = async (): Promise<GetConfigResponse> => {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}config/`, {
        method: "GET",
        credentials: "same-origin",
    });
    return await (r.json() as Promise<GetConfigResponse>);
}