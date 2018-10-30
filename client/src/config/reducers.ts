import { AllActions } from "../actions";
import * as browserActions from '../browser/actions';
import * as datasetActions from '../dataset/actions';
import { DatasetFormParams, MsgPartConfig } from "../messages";
import * as configActions from './actions';

export type ConfigState = MsgPartConfig & {
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
            return Object.assign({}, state, {
                lastOpened: newLastOpened,
            });
        }
    }
    return state;
}