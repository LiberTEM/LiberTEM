import { getApiBasePath } from "../helpers/apiHelpers";
import { DirectoryListingResponse, StatResponse } from "../messages";

export const getDirectoryListing = async (path: string): Promise<DirectoryListingResponse> => {
    const basePath = getApiBasePath();
    const url = `${basePath}browse/localfs/?path=${encodeURIComponent(path)}`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await (r.json() as Promise<DirectoryListingResponse>);
}

export const getPathStat = async (path: string): Promise<StatResponse> => {
    const basePath = getApiBasePath();
    const url = `${basePath}browse/localfs/stat/?path=${encodeURIComponent(path)}`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await (r.json() as Promise<StatResponse>);
}