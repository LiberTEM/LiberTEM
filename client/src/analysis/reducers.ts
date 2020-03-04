import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, filterWithPred, insertById, updateById } from "../helpers/reducerHelpers";
import * as jobActions from '../job/actions';
import * as analysisActions from "./actions";
import { AnalysisState } from "./types";


export type AnalysisReducerState = ById<AnalysisState>;

const initialAnalysisState: AnalysisReducerState = {
    byId: {},
    ids: [],
}

export function analysisReducer(state = initialAnalysisState, action: AllActions): AnalysisReducerState {
    switch (action.type) {
        case analysisActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.analysis.id, action.payload.analysis);
        }
        case analysisActions.ActionTypes.REMOVED: {
            return filterWithPred(state, (r: AnalysisState) => r.id !== action.payload.id);
        }
        case analysisActions.ActionTypes.UPDATE: {
            return updateById(state, action.payload.id, {
                details: action.payload.parameters,
            });
        }
        case jobActions.ActionTypes.CREATE: {
            return state; // FIXME: add job to appropriate analysis?
        }
        case channelActions.ActionTypes.TASK_RESULT: {
            const analysisIdForJob = state.ids.find(id => {
                const analysis = state.byId[id];
                return analysis.jobs.some(job => job === action.payload.job)
            });
            if (!analysisIdForJob) {
                return state;
            }
            return updateById(state, analysisIdForJob, {
                displayedJob: action.payload.job,
            });
        }
    }
    return state;
}
