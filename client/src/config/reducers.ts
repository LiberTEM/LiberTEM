import { AllActions } from "../actions";
import * as browserActions from '../browser/actions';
import { MsgPartConfig } from "../messages";
import * as configActions from './actions';

export type ConfigState = MsgPartConfig;

const initialConfigState = {
    version: "",
    localCores: 0,
    cwd: "/",
    separator: "/",
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
    }
    return state;
}