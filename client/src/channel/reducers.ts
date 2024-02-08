import { AllActions } from "../actions";
import * as channelActions from './actions';

export interface ChannelStatusReducer {
    status: ChannelStatusCodes,
}

export enum ChannelStatusCodes {
    CONNECTING = "connecting",
    CONNECTED = "connected",
    SNOOZED = "snoozed",
    UNSNOOZING = "unsnoozing",
    READY = "ready",
    WAITING = "waiting",
    DISCONNECTED = "disconnected"
}

const initialChannelState: ChannelStatusReducer = {
    status: ChannelStatusCodes.WAITING,
}

export const channelStatusReducer = (state = initialChannelState, action: AllActions): ChannelStatusReducer => {
    switch (action.type) {
        case channelActions.ActionTypes.OPEN: {
            return { status: ChannelStatusCodes.CONNECTED };
        }
        case channelActions.ActionTypes.INITIAL_STATE: {
            return { status: ChannelStatusCodes.READY };
        }
        case channelActions.ActionTypes.CLOSE: {
            return { status: ChannelStatusCodes.WAITING };
        }
        case channelActions.ActionTypes.SNOOZE: {
            return { status: ChannelStatusCodes.SNOOZED };
        }
        case channelActions.ActionTypes.UNSNOOZE: {
            return { status: ChannelStatusCodes.UNSNOOZING };
        }
        case channelActions.ActionTypes.UNSNOOZE_DONE: {
            return { status: ChannelStatusCodes.READY };
        }        
        case channelActions.ActionTypes.SHUTDOWN: {
            return { status: ChannelStatusCodes.DISCONNECTED }
        }
    }
    return state;
}