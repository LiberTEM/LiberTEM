import { AllActions } from "../actions";
import * as configActions from '../config/actions';
import { DirectoryListingDetails, FSPlace } from "../messages";
import * as browserActions from './actions';
import { DirectoryBrowserState } from "./types";

const initialBrowserState: DirectoryBrowserState = {
    isOpen: false,
    isOpenStack: false,
    isLoading: true,
    path: "/",
    drives: [],
    places: {},
    files: [] as DirectoryListingDetails[],
    dirs: [] as DirectoryListingDetails[],
}

export function directoryBrowserReducer(state: DirectoryBrowserState = initialBrowserState, action: AllActions): DirectoryBrowserState {
    switch (action.type) {
        case browserActions.ActionTypes.TOGGLE_FILE:{
          return Object.assign({}, state, {
            files: state.files.map((file, i) => {
              if ( i === action.payload.index) {
                return Object.assign({}, file, {
                  checked: !file.checked,
                });
              }
              return file;
            })
          })
        }
        case browserActions.ActionTypes.SELECT_ALL:{
          return Object.assign({},state,{
            files: state.files.map((file) => {
                return Object.assign({}, file, {
                  checked: true,
                });
            })
          })
        }
        case configActions.ActionTypes.FETCHED: {
            return Object.assign({}, state, {
                path: action.payload.config.cwd,
            })
        }
        case browserActions.ActionTypes.OPEN: {
            return Object.assign({}, state, {
                isOpen: true,
            })
        }
        case browserActions.ActionTypes.TOGGLE_STACK: {
            return Object.assign({}, state, {
                isOpenStack: !state.isOpenStack,
            })
        }
        case browserActions.ActionTypes.CANCEL: {
            return Object.assign({}, state, {
                isOpen: false,
            })
        }
        case browserActions.ActionTypes.LIST_DIRECTORY: {
            return Object.assign({}, state, {
                isLoading: true,
            })
        }
        case browserActions.ActionTypes.DIRECTORY_LISTING: {
            const places = action.payload.places.reduce((acc, place: FSPlace) => {
                return Object.assign({}, acc, {
                    [place.key]: place,
                })
            }, {});
            return Object.assign({}, state, {
                isLoading: false,
                path: action.payload.path,
                files: action.payload.files,
                dirs: action.payload.dirs,
                drives: action.payload.drives,
                places,
            })
        }
        case browserActions.ActionTypes.SELECT_FULL_PATH:
        case browserActions.ActionTypes.SELECT: {
            return Object.assign({}, state, {
                isLoading: false,
                isOpen: false,
            });
        }
        case browserActions.ActionTypes.SELECT_FILES: {
            return Object.assign({}, state, {
                isLoading: false,
                isOpen: false,
            });
        }
    }
    return state;
}