import { AllActions } from "../actions";
import { ConnectRequestParams } from "../messages";
import * as clusterActions from './actions';

export type ClusterConnectionState = {
    status: "connected",
    params: ConnectRequestParams
} | {
    status: "disconnected",
} | {
    status: "unknown"
} | {
    status: "connecting"
} | {
    status: "snoozed"
} | {
    status: "unsnoozing"
}

const initialClusterConnectionState: ClusterConnectionState = {
    status: "unknown"
}

export const clusterConnectionReducer = (state: ClusterConnectionState = initialClusterConnectionState, action: AllActions): ClusterConnectionState => {
    switch (action.type) {
        case clusterActions.ActionTypes.NOT_CONNECTED: {
            return {
                status: "disconnected"
            };
        }
        case clusterActions.ActionTypes.CONNECTED: {
            return {
                status: "connected",
                params: action.payload.params,
            }
        }
        case clusterActions.ActionTypes.SNOOZE: {
            return {
                status: "snoozed",
            }
        }
        case clusterActions.ActionTypes.UNSNOOZING: {
            return {
                status: "unsnoozing",
            }
        }        
        case clusterActions.ActionTypes.CONNECTING: {
            return {
                status: "connecting"
            }
        }
    }
    return state;
}