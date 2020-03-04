import { AllActions } from "../actions";
import * as datasetActions from "../dataset/actions";
import { ById, filterWithPred, insertById } from "../helpers/reducerHelpers";
import * as compoundAnalysisActions from "./actions";
import { CompoundAnalysisState } from "./types";

export type CompoundAnalysisReducerState = ById<CompoundAnalysisState>;

const initialCompoundAnalysisState: CompoundAnalysisReducerState = {
    byId: {},
    ids: [],
}

export function compoundAnalysisReducer(state = initialCompoundAnalysisState, action: AllActions): CompoundAnalysisReducerState {
    switch (action.type) {
        case compoundAnalysisActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.analysis.id, action.payload.analysis);
        }
        case compoundAnalysisActions.ActionTypes.REMOVED: {
            return filterWithPred(state, (r: CompoundAnalysisState) => r.id !== action.payload.id);
        }
        case datasetActions.ActionTypes.DELETE: {
            return filterWithPred(state, (r: CompoundAnalysisState) => r.dataset !== action.payload.dataset);
        }
    }
    return state;
}
