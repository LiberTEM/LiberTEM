import { AllActions } from "../actions";
import * as analysisActions from '../analysis/actions';
import * as channelActions from '../channel/actions';
import { ById, insertById, updateById } from "../helpers/reducerHelpers";
import { JobResultType, JobState } from "./types";

export type JobReducerState = ById<JobState>;

const initialJobState: JobReducerState = {
    byId: {},
    ids: [],
};

export function jobReducer(state = initialJobState, action: AllActions) {
    switch (action.type) {
        case analysisActions.ActionTypes.RUNNING: {
            // in case there is no job record yet for the job id, 
            const currentJob = state.byId[action.payload.job];
            if (currentJob === undefined) {
                return insertById(
                    state,
                    action.payload.job,
                    {
                        id: action.payload.job,
                        results: [] as JobResultType[],
                        running: "CREATING",
                        status: "CREATING",
                    }
                )
            } else {
                return state;
            }
        }
        case channelActions.ActionTypes.START_JOB: {
            return insertById(
                state,
                action.payload.job,
                {
                    id: action.payload.job,
                    dataset: action.payload.dataset,
                    results: [] as JobResultType[],
                    running: "RUNNING",
                    status: "IN_PROGRESS",
                    startTimestamp: action.payload.timestamp,
                }
            )
        }
        case channelActions.ActionTypes.TASK_RESULT: {
            return updateById(
                state,
                action.payload.job,
                {
                    results: action.payload.results,
                }
            );
        }
        case channelActions.ActionTypes.FINISH_JOB: {
            return updateById(
                state,
                action.payload.job,
                {
                    results: action.payload.results,
                    running: "DONE",
                    status: "SUCCESS",
                    endTimestamp: action.payload.timestamp,
                }
            );
        }
        case channelActions.ActionTypes.JOB_ERROR: {
            return updateById(
                state,
                action.payload.job,
                {
                    running: "DONE",
                    status: "ERROR",
                    endTimestamp: action.payload.timestamp,
                }
            )
        }
    }
    return state;
}