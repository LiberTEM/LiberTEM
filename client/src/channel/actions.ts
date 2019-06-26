import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { JobResultType } from '../job/types';
import { MsgPartInitialDataset, MsgPartJob } from '../messages';


export type PartialResultType = JobResultType;

export enum ActionTypes {
    INITIAL_STATE = 'CHANNEL_INITIAL_STATE',
    JOB_STARTED = 'CHANNEL_JOB_STARTED',
    FINISH_JOB = 'CHANNEL_FINISH_JOB',
    TASK_RESULT = 'CHANNEL_TASK_RESULT',
    JOB_ERROR = 'CHANNEL_JOB_ERROR',
    OPEN = "CHANNEL_OPEN",
    CLOSE = "CHANNEL_CLOSE",
    ERROR = "CHANNEL_ERROR",
    CANCELLED = "CANCELLED",
}

export const Actions = {
    initialState: (jobs: MsgPartJob[], datasets: MsgPartInitialDataset[], timestamp: number) => createAction(ActionTypes.INITIAL_STATE, { jobs, datasets, timestamp }),
    jobStarted: (job: string, dataset: string, timestamp: number) => createAction(ActionTypes.JOB_STARTED, { job, timestamp, dataset }),
    finishJob: (job: string, results: JobResultType[], timestamp: number) => createAction(ActionTypes.FINISH_JOB, { job, results, timestamp }),
    taskResult: (job: string, results: PartialResultType[], timestamp: number) => createAction(ActionTypes.TASK_RESULT, { job, results, timestamp }),
    jobError: (job: string, msg: string, id: string, timestamp: number) => createAction(ActionTypes.JOB_ERROR, { job, msg, id, timestamp }),
    open: (timestamp: number) => createAction(ActionTypes.OPEN, { timestamp }),
    close: (timestamp: number) => createAction(ActionTypes.CLOSE, { timestamp }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
    cancelled: (job: string) => createAction(ActionTypes.CANCELLED, { job }),
}

export type Actions = ActionsUnion<typeof Actions>;