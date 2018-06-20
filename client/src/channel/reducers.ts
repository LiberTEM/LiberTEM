import { AllActions } from "../actions";
import * as channelActions from './actions';

interface ChannelState {
    status: "connecting" | "connected" | "ready" | "waiting"
}

const initialChannelState: ChannelState = {
    status: "waiting",
}

export function channelStatusReducer(state = initialChannelState, action: AllActions) {
    switch (action.type) {
        case channelActions.ActionTypes.OPEN: {
            return { status: "connected" };
        }
        case channelActions.ActionTypes.INITIAL_STATE: {
            return { status: "ready" };
        }
        case channelActions.ActionTypes.CLOSE: {
            return { status: "waiting" };
        }
    }
    return state;
}