import { getApiBasePath } from "../helpers/apiHelpers";
import { DirectoryListingResponse } from "../messages";

export const getDirectoryListing = async (path: string): Promise<DirectoryListingResponse> => {
    const basePath = getApiBasePath();
    const url = `${basePath}browse/localfs/?path=${encodeURIComponent(path)}`;
    const r = await fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    });
    return await (r.json() as Promise<DirectoryListingResponse>);
}