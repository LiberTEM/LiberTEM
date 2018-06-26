import { AllActions } from "../actions";
import { ById, insertById, updateById } from "../helpers/reducerHelpers";
import * as analysisActions from "./actions";
import { Analysis } from "./types";

export type AnalysisReducerState = ById<Analysis>;

const initialAnalysisState: AnalysisReducerState = {
    byId: {},
    ids: [],
}

export function analysisReducer(state = initialAnalysisState, action: AllActions) {
    switch (action.type) {
        case analysisActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.analysis.id, action.payload.analysis);
        }
        case analysisActions.ActionTypes.UPDATE_PARAMETERS: {
            const details = state.byId[action.payload.id].details;
            const newDetails = Object.assign({}, details, {
                parameters: Object.assign({}, details.parameters, action.payload.parameters),
            })
            return updateById(state, action.payload.id, {
                details: newDetails,
            });
        }
        case analysisActions.ActionTypes.RUNNING: {
            return updateById(state, action.payload.id, { currentJob: action.payload.job })
        }
    }
    return state;
}