import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, getById, insertById, updateById } from "../helpers/reducerHelpers";
import { Job, JobResultType } from "./types";

export type JobReducerState = ById<Job>;

const initialJobState : JobReducerState = {
    byId: {},
    ids: [],
};

export function jobReducer(state = initialJobState, action: AllActions) {
    switch(action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            const jobs = action.payload.jobs.map(job => ({
                dataset: job.dataset,
                id: job.job,
                results: ([] as JobResultType[]),
                // TODO: real status here
                running: "DONE",
                status: "SUCCESS",
            }))
            return {
                byId: getById(jobs, job => job.id),
                ids: jobs.map(job => job.id)
            };
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
                }
            );
        }
    }
    return state;
}