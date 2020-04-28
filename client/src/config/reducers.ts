import { AllActions } from "../actions";
import * as browserActions from '../browser/actions';
import * as clusterActions from '../cluster/actions'
import * as datasetActions from '../dataset/actions';
import { ClusterTypes, DatasetFormParams, MsgPartConfig } from "../messages";
import * as configActions from './actions';
import { makeUnique } from "./helpers";

export interface LocalConfig {
    cwd: string,
    fileHistory: string[],
    lastOpened: {
        [path: string]: DatasetFormParams
    },
    lastConnection: {
        type: ClusterTypes,
        address: string
    }
}

export type ConfigParams = MsgPartConfig & LocalConfig ;
export type ConfigState = ConfigParams & {
    haveConfig: boolean,
};

const initialConfigState: ConfigState = {
    version: "",
    revision: "",
    localCores: 0,
    cwd: "/",
    separator: "/",
    lastOpened: {},
    resultFileFormats: {},
    fileHistory: [],
    haveConfig: false,
    lastConnection: {
        type: ClusterTypes.LOCAL,
        address: "tcp://localhost:8786",
    }
}

export function configReducer(state = initialConfigState, action: AllActions): ConfigState {
    switch (action.type) {
        case configActions.ActionTypes.FETCHED: {
            return Object.assign({}, action.payload.config, { haveConfig: true });
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
        case clusterActions.ActionTypes.CONNECT: {
            if (action.payload.params.type === ClusterTypes.LOCAL){
                const newLastConnection = Object.assign({}, state.lastConnection, {type: ClusterTypes.LOCAL})
                return Object.assign({}, state, {
                    lastConnection : newLastConnection
                })
            }
            else {
                return Object.assign({}, state, {
                    lastConnection: action.payload.params
                })
            }
        }
    }
    return state;
}