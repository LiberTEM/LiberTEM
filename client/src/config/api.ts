import { GetConfigResponse } from "../messages";

export function getConfig(): Promise<GetConfigResponse> {
    return fetch(`/api/config/`, {
        method: "GET",
        credentials: "same-origin",
    }).then(r => r.json());
}