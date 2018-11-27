import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { DatasetFormParams, DirectoryListingDetails } from "../messages";

export enum ActionTypes {
    LIST_DIRECTORY = 'BROWSER_LIST_DIRECTORY',
    DIRECTORY_LISTING = 'BROWSER_DIRECTORY_LISTING',
    DETECT_PARAMS = 'BROWSER_DETECT_PARAMS',
    PARAMS_DETECTED = 'BROWSER_PARAMS_DETECTED',
    ERROR = 'BROWSER_ERROR',
    OPEN = 'BROWSER_OPEN',
    CANCEL = 'BROWSER_CANCEL',
    SELECT = 'BROWSER_SELECT',
    SELECT_FULL_PATH = 'BROWSER_SELECT_FULL_PATH',
}

export const Actions = {
    list: (path: string, name?: string) => createAction(ActionTypes.LIST_DIRECTORY, { path, name }),
    dirListing: (path: string, dirs: DirectoryListingDetails[], files: DirectoryListingDetails[]) => createAction(ActionTypes.DIRECTORY_LISTING, { path, dirs, files }),
    open: () => createAction(ActionTypes.OPEN),
    cancel: () => createAction(ActionTypes.CANCEL),
    select: (path: string, name: string) => createAction(ActionTypes.SELECT, { path, name }),
    selectFullPath: (path: string) => createAction(ActionTypes.SELECT_FULL_PATH, { path }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
    detectParams: (path: string) => createAction(ActionTypes.DETECT_PARAMS, { path }),
    paramsDetected: (path: string, params: DatasetFormParams) => createAction(ActionTypes.PARAMS_DETECTED, { path, params }),
}

export type Actions = ActionsUnion<typeof Actions>;