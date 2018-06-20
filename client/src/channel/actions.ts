import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { JobResultType } from '../job/types';
import { MsgPartDataset, MsgPartJob } from './messages';


export type PartialResultType = JobResultType;

export enum ActionTypes {
    INITIAL_STATE = 'CHANNEL_INITIAL_STATE',
    START_JOB = 'CHANNEL_START_JOB',
    FINISH_JOB = 'CHANNEL_FINISH_JOB',
    TASK_RESULT = 'CHANNEL_TASK_RESULT',
    OPEN = "OPEN",
    CLOSE = "CLOSE",
    ERROR = "ERROR",
}

export const Actions = {
    // tslint:disable object-literal-sort-keys
    initialState: (jobs: MsgPartJob[], datasets: MsgPartDataset[]) => createAction(ActionTypes.INITIAL_STATE, { jobs, datasets }),
    startJob: (job: string) => createAction(ActionTypes.START_JOB, { job }),
    finishJob: (job: string, results: JobResultType[]) => createAction(ActionTypes.FINISH_JOB, { job, results }),
    taskResult: (job: string, results: PartialResultType[]) => createAction(ActionTypes.TASK_RESULT, { job, results }),
    open: () => createAction(ActionTypes.OPEN),
    close: () => createAction(ActionTypes.CLOSE),
    error: (msg: string) => createAction(ActionTypes.ERROR, { msg }),
    // tslint:enable
}

export type Actions = ActionsUnion<typeof Actions>;