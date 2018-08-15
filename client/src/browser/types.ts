import { DirectoryListingDetails } from "../messages";

export interface DirectoryBrowserState {
    isOpen: boolean,
    isLoading: boolean,
    path: string,
    files: DirectoryListingDetails[],
    dirs: DirectoryListingDetails[],
}