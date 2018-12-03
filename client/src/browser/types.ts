import { DirectoryListingDetails, FSPlace } from "../messages";

export interface DirectoryBrowserState {
    isOpen: boolean,
    isLoading: boolean,
    path: string,
    drives: string[],
    places: FSPlace[],
    files: DirectoryListingDetails[],
    dirs: DirectoryListingDetails[],
}