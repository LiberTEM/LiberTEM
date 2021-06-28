import { AllActions } from "../actions";
import * as browserActions from '../browser/actions';
import * as channelActions from '../channel/actions';
import * as clusterActions from '../cluster/actions';
import * as analysisActions from '../compoundAnalysis/actions';
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

export const errorReducer = (state = initialErrorState, action: AllActions): ErrorState => {
    switch (action.type) {
        case browserActions.ActionTypes.ERROR:
        case datasetActions.ActionTypes.ERROR:
        case channelActions.ActionTypes.ERROR:
        case clusterActions.ActionTypes.ERROR:
        case analysisActions.ActionTypes.ERROR:
        case errorActions.ActionTypes.GENERIC:
        case channelActions.ActionTypes.JOB_ERROR: {
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
        case errorActions.ActionTypes.DISMISS_ALL: {
            return initialErrorState;
        }
    }
    return state;
}