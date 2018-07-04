import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { JobResultType } from '../job/types';
import { MsgPartDataset, MsgPartJob } from '../messages';


export type PartialResultType = JobResultType;

export enum ActionTypes {
    INITIAL_STATE = 'CHANNEL_INITIAL_STATE',
    START_JOB = 'CHANNEL_START_JOB',
    FINISH_JOB = 'CHANNEL_FINISH_JOB',
    TASK_RESULT = 'CHANNEL_TASK_RESULT',
    OPEN = "CHANNEL_OPEN",
    CLOSE = "CHANNEL_CLOSE",
    ERROR = "CHANNEL_ERROR",
}

export const Actions = {
    initialState: (jobs: MsgPartJob[], datasets: MsgPartDataset[], timestamp: number) => createAction(ActionTypes.INITIAL_STATE, { jobs, datasets, timestamp }),
    startJob: (job: string, timestamp: number) => createAction(ActionTypes.START_JOB, { job, timestamp }),
    finishJob: (job: string, results: JobResultType[], timestamp: number) => createAction(ActionTypes.FINISH_JOB, { job, results, timestamp }),
    taskResult: (job: string, results: PartialResultType[], timestamp: number) => createAction(ActionTypes.TASK_RESULT, { job, results, timestamp }),
    open: (timestamp: number) => createAction(ActionTypes.OPEN, { timestamp }),
    close: (timestamp: number) => createAction(ActionTypes.CLOSE, { timestamp }),
    error: (msg: string, timestamp: number) => createAction(ActionTypes.ERROR, { msg, timestamp }),
}

export type Actions = ActionsUnion<typeof Actions>;