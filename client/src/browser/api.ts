import { getApiBasePath } from "../helpers/apiHelpers";
import { DirectoryListingResponse } from "../messages";

export function getDirectoryListing(path: string): Promise<DirectoryListingResponse> {
    const basePath = getApiBasePath();
    const url = `${basePath}browse/localfs/?path=${encodeURIComponent(path)}`;
    return fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    }).then(r => r.json());
}