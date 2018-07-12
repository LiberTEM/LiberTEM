import { AllActions } from "../actions";
import * as analysisActions from '../analysis/actions';
import * as channelActions from '../channel/actions';
import * as datasetActions from '../dataset/actions';
import { ById, filterWithPred, insertById } from "../helpers/reducerHelpers";
import * as errorActions from './actions';

export interface ErrorMessage {
    id: string,
    msg: string,
    timestamp: number,
}

export type ErrorState = ById<ErrorMessage>;

const initialErrorState: ErrorState = {
    byId: {},
    ids: [],
};

export function errorReducer(state = initialErrorState, action: AllActions): ErrorState {
    switch (action.type) {
        case datasetActions.ActionTypes.ERROR:
        case channelActions.ActionTypes.ERROR:
        case analysisActions.ActionTypes.ERROR: {
            return insertById(state, action.payload.id, {
                id: action.payload.id,
                msg: action.payload.msg,
                timestamp: action.payload.timestamp,
            });
        }
        case channelActions.ActionTypes.OPEN: {
            return initialErrorState;
        }
        case errorActions.ActionTypes.DISMISS: {
            return filterWithPred(state, (r: ErrorMessage) => r.id !== action.payload.id);
        }
    }
    return state;
}