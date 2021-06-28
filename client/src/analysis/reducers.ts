import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, constructById, filterWithPred, insertById, updateById, updateWithMap } from "../helpers/reducerHelpers";
import * as jobActions from '../job/actions';
import * as analysisActions from "./actions";
import { AnalysisState } from "./types";


export type AnalysisReducerState = ById<AnalysisState>;

const initialAnalysisState: AnalysisReducerState = {
    byId: {},
    ids: [],
}

export const analysisReducer = (state = initialAnalysisState, action: AllActions): AnalysisReducerState => {
    switch (action.type) {
        case analysisActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.analysis.id, action.payload.analysis);
        }
        case analysisActions.ActionTypes.REMOVED: {
            return filterWithPred(state, (r: AnalysisState) => r.id !== action.payload.id);
        }
        case analysisActions.ActionTypes.UPDATED: {
            return updateById(state, action.payload.id, {
                details: action.payload.details,
            });
        }
        case jobActions.ActionTypes.CREATE: {
            const analysis = state.byId[action.payload.analysis];
            const oldJobs = analysis.jobs ? analysis.jobs : [];
            return updateById(state, action.payload.analysis, {
                jobs: [action.payload.id, ...oldJobs],
            })
        }
        case channelActions.ActionTypes.CANCEL_JOB_FAILED:
        case channelActions.ActionTypes.CANCELLED: {
            // remove job from the matching analysis
            return updateWithMap(state, (analysis) => ({
                ...analysis,
                jobs: analysis.jobs.filter((job) => job !== action.payload.job),
            }));
        }
        case channelActions.ActionTypes.INITIAL_STATE: {
            const analysisState: AnalysisState[] = action.payload.analyses.map(item => ({
                doAutoStart: false,
                id: item.analysis,
                dataset: item.dataset,
                details: item.details,
                // FIXME: add jobs!
                jobs: item.jobs,
            }));
            return {
                byId: constructById(analysisState, analysis => analysis.id),
                ids: action.payload.analyses.map(analysis => analysis.analysis),
            }
        }
        case channelActions.ActionTypes.FINISH_JOB:
        case channelActions.ActionTypes.TASK_RESULT: {
            const analysisIdForJob = state.ids.find(id => {
                const analysis = state.byId[id];
                const jobs = analysis.jobs ? analysis.jobs : [];
                return jobs.some(job => job === action.payload.job)
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
