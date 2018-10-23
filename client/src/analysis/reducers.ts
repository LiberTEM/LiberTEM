import { AllActions } from "../actions";
import * as datasetActions from "../dataset/actions";
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
            // TODO: convince typescript that `[key]: newDetails` is a better way...
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
            const { kind, id } = action.payload;
            const analysis = state.byId[id];
            const oldJob = analysis.jobs[kind];
            let jobHistory = analysis.jobHistory;
            if (oldJob !== undefined) {
                // TODO: length restriction?
                jobHistory = Object.assign({}, jobHistory, {
                    [kind]: [oldJob, ...jobHistory[kind]],
                });
            }
            const newJobs: JobList = Object.assign({}, analysis.jobs, {
                [action.payload.kind]: action.payload.job,
            });
            return updateById(state, action.payload.id, { jobs: newJobs, jobHistory })
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
        case datasetActions.ActionTypes.DELETE: {
            return filterWithPred(state, (r: AnalysisState) => r.dataset !== action.payload.dataset);
        }
    }
    return state;
}