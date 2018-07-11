import { AllActions } from "../actions";
import { MsgPartConfig } from "../messages";
import * as configActions from './actions';

export type ConfigState = MsgPartConfig;

const initialConfigState = {
    version: "",
    localCores: 0,
}

export function configReducer(state = initialConfigState, action: AllActions) {
    switch (action.type) {
        case configActions.ActionTypes.FETCHED: {
            return action.payload.config;
        }
    }
    return state;
}