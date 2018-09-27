import { AllActions } from "../actions";
import { ById, filterWithPred, insertById, updateById } from "../helpers/reducerHelpers";
import * as analysisActions from "./actions";
import { AnalysisState, JobList } from "./types";

export type AnalysisReducerState = ById<AnalysisState>;

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
            const key = action.payload.kind === "FRAME" ? "frameDetails" : "resultDetails";
            const details = state.byId[action.payload.id][key];
            const newDetails = Object.assign({}, details, {
                parameters: Object.assign({}, details.parameters, action.payload.parameters),
            })
            // TODO: find generic way
            if (action.payload.kind === "FRAME") {
                return updateById(state, action.payload.id, {
                    frameDetails: newDetails,
                });
            } else {
                return updateById(state, action.payload.id, {
                    resultDetails: newDetails,
                });
            }
        }
        case analysisActions.ActionTypes.RUNNING: {
            const newJobs: JobList = Object.assign({}, state.byId[action.payload.id].jobs, {
                [action.payload.kind]: action.payload.job,
            });
            return updateById(state, action.payload.id, { jobs: newJobs })
        }
        case analysisActions.ActionTypes.REMOVED: {
            return filterWithPred(state, (r: AnalysisState) => r.id !== action.payload.id);
        }
        case analysisActions.ActionTypes.SET_FRAMEVIEW_MODE: {
            const newFrameDetails = Object.assign({}, state.byId[action.payload.id].frameDetails, {
                type: action.payload.mode,
                parameters: action.payload.initialParams,
            });
            return updateById(state, action.payload.id, { frameDetails: newFrameDetails });
        }
    }
    return state;
}