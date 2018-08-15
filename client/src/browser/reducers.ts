import { AllActions } from "../actions";
import * as configActions from '../config/actions';
import { DirectoryListingDetails } from "../messages";
import * as browserActions from './actions';
import { DirectoryBrowserState } from "./types";

const initialBrowserState: DirectoryBrowserState = {
    isOpen: false,
    isLoading: true,
    path: "/",
    files: [] as DirectoryListingDetails[],
    dirs: [] as DirectoryListingDetails[],
}

export function directoryBrowserReducer(state: DirectoryBrowserState = initialBrowserState, action: AllActions): DirectoryBrowserState {
    switch (action.type) {
        case configActions.ActionTypes.FETCHED: {
            return Object.assign({}, state, {
                path: action.payload.config.cwd,
            })
            break;
        }
        case browserActions.ActionTypes.OPEN: {
            return Object.assign({}, state, {
                isOpen: true,
            })
            break;
        }
        case browserActions.ActionTypes.CANCEL: {
            return Object.assign({}, state, {
                isOpen: false,
            })
            break;
        }
        case browserActions.ActionTypes.LIST_DIRECTORY: {
            return Object.assign({}, state, {
                isLoading: true,
            })
            break;
        }
        case browserActions.ActionTypes.DIRECTORY_LISTING: {
            return Object.assign({}, state, {
                isLoading: false,
                path: action.payload.path,
                files: action.payload.files,
                dirs: action.payload.dirs,
            })
            break;
        }
        case browserActions.ActionTypes.SELECT: {
            return Object.assign({}, state, {
                isLoading: false,
                isOpen: false,
            });
        }
    }
    return state;
}