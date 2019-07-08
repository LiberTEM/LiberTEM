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

export function analysisReducer(state = initialAnalysisState, action: AllActions): AnalysisReducerState {
    switch (action.type) {
        case analysisActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.analysis.id, action.payload.analysis);
        }
        case analysisActions.ActionTypes.PREPARE_RUN: {
            const { jobIndex, id } = action.payload;
            const analysis = state.byId[id];
            const oldJob = analysis.jobs[jobIndex];
            const jobHistory = [...analysis.jobHistory];
            if (oldJob !== undefined) {
                // TODO: length restriction?
                const hist = jobHistory[jobIndex] ? jobHistory[jobIndex] : [];
                jobHistory[jobIndex] = [oldJob, ...hist];
            }
            const newJobs: JobList = [...analysis.jobs];
            newJobs[jobIndex] = action.payload.job;
            return updateById(state, action.payload.id, { jobs: newJobs, jobHistory })
        }
        case analysisActions.ActionTypes.REMOVED: {
            return filterWithPred(state, (r: AnalysisState) => r.id !== action.payload.id);
        }
        case datasetActions.ActionTypes.DELETE: {
            return filterWithPred(state, (r: AnalysisState) => r.dataset !== action.payload.dataset);
        }
    }
    return state;
}