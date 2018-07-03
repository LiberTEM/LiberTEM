import { AllActions } from "../actions";
import * as analysisActions from '../analysis/actions';
import * as channelActions from '../channel/actions';
import { ById, constructById, insertById, updateById } from "../helpers/reducerHelpers";
import { JobResultType, JobState } from "./types";

export type JobReducerState = ById<JobState>;

const initialJobState: JobReducerState = {
    byId: {},
    ids: [],
};

export function jobReducer(state = initialJobState, action: AllActions) {
    switch (action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            const jobs = action.payload.jobs.map(job => ({
                dataset: job.dataset,
                id: job.id,
                results: ([] as JobResultType[]),
                // TODO: real status here
                running: "DONE",
                status: "SUCCESS",
            }))
            return {
                byId: constructById(jobs, job => job.id),
                ids: jobs.map(job => job.id)
            };
        }
        case analysisActions.ActionTypes.RUNNING: {
            // in case there is no job record yet for the job id, 
            const currentJob = state.byId[action.payload.job];
            if (currentJob === undefined) {
                return insertById(
                    state,
                    action.payload.job,
                    {
                        id: action.payload.job,
                        results: ([] as JobResultType[]),
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
                    results: ([] as JobResultType[]),
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
    }
    return state;
}