import { DirectoryListingDetails, FSPlace } from "../messages";

export interface FSPlaces {
    [key: string]: FSPlace,
}

export interface DirectoryBrowserState {
    isOpen: boolean,
    isLoading: boolean,
    isOpenStack: boolean,
    path: string,
    drives: string[],
    places: FSPlaces
    files: DirectoryListingDetails[],
    dirs: DirectoryListingDetails[],
}