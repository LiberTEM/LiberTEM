import { AllActions } from "../actions";
import * as browserActions from '../browser/actions';
import * as datasetActions from '../dataset/actions';
import { DatasetFormParams, MsgPartConfig } from "../messages";
import * as configActions from './actions';
import { makeUnique } from "./helpers";

export type ConfigState = MsgPartConfig & {
    fileHistory: string[],
    lastOpened: {
        [path: string]: DatasetFormParams
    }
};

const initialConfigState: ConfigState = {
    version: "",
    revision: "",
    localCores: 0,
    cwd: "/",
    separator: "/",
    lastOpened: {},
    fileHistory: [],
}

export function configReducer(state = initialConfigState, action: AllActions) {
    switch (action.type) {
        case configActions.ActionTypes.FETCHED: {
            return action.payload.config;
        }
        case browserActions.ActionTypes.DIRECTORY_LISTING: {
            return Object.assign({}, state, {
                cwd: action.payload.path,
            });
        }
        case datasetActions.ActionTypes.CREATE: {
            const newLastOpened = Object.assign({}, state.lastOpened, { [action.payload.dataset.params.path]: action.payload.dataset.params });
            const newFileHistory = makeUnique([
                action.payload.dataset.params.path, ...state.fileHistory
            ]).slice(0, 11);
            return Object.assign({}, state, {
                lastOpened: newLastOpened,
                fileHistory: newFileHistory,
            });
        }
    }
    return state;
}