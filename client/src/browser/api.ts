import { DirectoryListingResponse } from "../messages";

export function getDirectoryListing(path: string): Promise<DirectoryListingResponse> {
    const url = `/api/browse/localfs/?path=${path}`;
    return fetch(url, {
        method: 'GET',
        credentials: "same-origin",
    }).then(r => r.json());
}