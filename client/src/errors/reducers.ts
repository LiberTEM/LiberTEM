import { AllActions } from "../actions";
import * as analysisActions from '../analysis/actions';
import * as channelActions from '../channel/actions';
import * as datasetActions from '../dataset/actions';

interface Error {
    msg: string,
    timestamp: number,
}

export interface ErrorState {
    errorList: Error[],
}

const initialErrorState: ErrorState = {
    errorList: [],
};

const insertError = (state: ErrorState, msg: string, timestamp: number): ErrorState => {
    return {
        errorList: [...state.errorList, { msg, timestamp }],
    };
}

export function errorReducer(state = initialErrorState, action: AllActions): ErrorState {
    switch (action.type) {
        case datasetActions.ActionTypes.ERROR:
        case channelActions.ActionTypes.ERROR:
        case analysisActions.ActionTypes.ERROR: {
            return insertError(state, action.payload.msg, action.payload.timestamp);
        }
    }
    return state;
}