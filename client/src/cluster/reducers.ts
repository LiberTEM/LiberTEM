import { AllActions } from "../actions";
import { ConnectRequestParams, HostDetails } from "../messages";
import * as clusterActions from './actions';

export type ClusterConnectionState = {
    status: "connected",
    params: ConnectRequestParams,
    details: HostDetails[]
} | {
    status: "disconnected",
} | {
    status: "unknown"
} | {
    status: "connecting"
}

const initialClusterConnectionState: ClusterConnectionState = {
    status: "unknown"
}

export function clusterConnectionReducer(state = initialClusterConnectionState, action: AllActions): ClusterConnectionState {
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
                details: action.payload.details,
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