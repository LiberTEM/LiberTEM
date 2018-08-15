import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { DirectoryListingDetails } from "../messages";

export enum ActionTypes {
    LIST_DIRECTORY = 'BROWSER_LIST_DIRECTORY',
    DIRECTORY_LISTING = 'BROWSER_DIRECTORY_LISTING',
    ERROR = 'BROWSER_ERROR',
    OPEN = 'BROWSER_OPEN',
    CANCEL = 'BROWSER_CANCEL',
    SELECT = 'BROWSER_SELECT',
}

export const Actions = {
    list: (path: string, name?: string) => createAction(ActionTypes.LIST_DIRECTORY, { path, name }),
    dirListing: (path: string, dirs: DirectoryListingDetails[], files: DirectoryListingDetails[]) => createAction(ActionTypes.DIRECTORY_LISTING, { path, dirs, files }),
    open: () => createAction(ActionTypes.OPEN),
    cancel: () => createAction(ActionTypes.CANCEL),
    select: (path: string, name: string) => createAction(ActionTypes.SELECT, { path, name }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;