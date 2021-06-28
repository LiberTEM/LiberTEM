import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, constructById, insertById, updateById } from "../helpers/reducerHelpers";
import * as jobActions from './actions';
import { JobRunning, JobState, JobStatus } from "./types";

export type JobReducerState = ById<JobState>;

const initialJobState: JobReducerState = {
    byId: {},
    ids: [],
};

export const jobReducer = (state = initialJobState, action: AllActions): JobReducerState => {
    switch (action.type) {
        case jobActions.ActionTypes.CREATE: {
            const createResult = insertById(
                state,
                action.payload.id,
                {
                    id: action.payload.id,
                    analysis: action.payload.analysis,
                    running: JobRunning.CREATING,
                    status: JobStatus.CREATING,
                    results: [],
                    startTimestamp: action.payload.timestamp,
                }
            )
            return createResult;
        }
        case channelActions.ActionTypes.JOB_STARTED: {
            return updateById(
                state,
                action.payload.job,
                {
                    running: JobRunning.RUNNING,
                    status: JobStatus.IN_PROGRESS,
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
            const { job, timestamp, results } = action.payload;
            return updateById(
                state,
                job,
                {
                    running: JobRunning.DONE,
                    status: JobStatus.SUCCESS,
                    results,
                    endTimestamp: timestamp,
                }
            );
        }
        case channelActions.ActionTypes.JOB_ERROR: {
            const { job, timestamp } = action.payload;
            return updateById(
                state,
                job,
                {
                    running: JobRunning.DONE,
                    status: JobStatus.ERROR,
                    endTimestamp: timestamp,
                }
            )
        }
        case channelActions.ActionTypes.INITIAL_STATE: {
            const jobs = action.payload.jobs;
            const jobState: JobState[] = jobs.map(job => ({
                    id: job.id,
                    analysis: job.analysis,
                    // FIXME: right job status!
                    status: JobStatus.SUCCESS,
                    startTimestamp: 0,
                    // FIXME: result blobs?
                    results: [],
                    // FIXME: right job running status!
                    running: JobRunning.DONE,
                    endTimestamp: 0,
            }));

            return {
                byId: constructById(jobState, job => job.id),
                ids: jobState.map(job => job.id),
            };
        }
    }
    return state;
}