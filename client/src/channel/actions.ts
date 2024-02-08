import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { JobResultType } from "../job/types";
import { AnalysisDetails, MsgPartAnalysis, MsgPartCompoundAnalysis, MsgPartInitialDataset, MsgPartJob, ProgressDetails } from "../messages";

export type PartialResultType = JobResultType;

export enum ActionTypes {
    INITIAL_STATE = "CHANNEL_INITIAL_STATE",
    JOB_STARTED = "CHANNEL_JOB_STARTED",
    JOB_PROGRESS = "JOB_PROGRESS",
    FINISH_JOB = "CHANNEL_FINISH_JOB",
    TASK_RESULT = "CHANNEL_TASK_RESULT",
    JOB_ERROR = "CHANNEL_JOB_ERROR",
    OPEN = "CHANNEL_OPEN",
    CLOSE = "CHANNEL_CLOSE",
    ERROR = "CHANNEL_ERROR",
    CANCELLED = "CANCELLED",
    SNOOZE = "SNOOZE",
    UNSNOOZE = "UNSNOOZE",
    UNSNOOZE_DONE = "UNSNOOZE_DONE",
    CANCEL_JOB_FAILED = "CANCEL_JOB_FAILED",
    SHUTDOWN = "CHANNEL_SHUTDOWN",
    CLOSE_LOOP = "CHANNEL_LOOP_CLOSE",
    ANALYSIS_CREATED = "ANALYSIS_CREATED",
    ANALYSIS_UPDATED = "ANALYSIS_UPDATED",
    ANALYSIS_REMOVED = "ANALYSIS_REMOVED",
}

export const Actions = {
    initialState: (
        jobs: MsgPartJob[],
        datasets: MsgPartInitialDataset[],
        compoundAnalyses: MsgPartCompoundAnalysis[],
        analyses: MsgPartAnalysis[],
        timestamp: number
    ) => createAction(ActionTypes.INITIAL_STATE, { jobs, datasets, timestamp, compoundAnalyses, analyses }),
    jobStarted: (job: string, timestamp: number) => createAction(ActionTypes.JOB_STARTED, { job, timestamp }),
    jobProgress: (job: string, details: ProgressDetails) => createAction(ActionTypes.JOB_PROGRESS, { job, details }),
    finishJob: (job: string, results: JobResultType[], timestamp: number) => createAction(ActionTypes.FINISH_JOB, { job, results, timestamp }),
    taskResult: (job: string, results: PartialResultType[], timestamp: number) => createAction(ActionTypes.TASK_RESULT, { job, results, timestamp }),
    jobError: (job: string, msg: string, id: string, timestamp: number) => createAction(ActionTypes.JOB_ERROR, { job, msg, id, timestamp }),
    open: (timestamp: number) => createAction(ActionTypes.OPEN, { timestamp }),
    close: (timestamp: number) => createAction(ActionTypes.CLOSE, { timestamp }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
    shutdown: (timestamp: number) => createAction(ActionTypes.SHUTDOWN, { timestamp }),
    closeloop: (timestamp: number) => createAction(ActionTypes.CLOSE_LOOP, { timestamp }),
    snooze: (timestamp: number) => createAction(ActionTypes.SNOOZE, { timestamp }),
    unsnooze: (timestamp: number) => createAction(ActionTypes.UNSNOOZE, { timestamp }),
    unsnooze_done: (timestamp: number) => createAction(ActionTypes.UNSNOOZE_DONE, { timestamp }),
    cancelled: (job: string) => createAction(ActionTypes.CANCELLED, { job }),
    cancelFailed: (job: string) => createAction(ActionTypes.CANCEL_JOB_FAILED, { job }),
    analysisCreated: (analysis: string, dataset: string, details: AnalysisDetails) =>
        createAction(ActionTypes.ANALYSIS_CREATED, { dataset, analysis, details }),
    analysisUpdated: (analysis: string, dataset: string, details: AnalysisDetails) =>
        createAction(ActionTypes.ANALYSIS_UPDATED, { dataset, analysis, details }),
    analysisRemoved: (analysis: string) => createAction(ActionTypes.ANALYSIS_REMOVED, { analysis }),
};

export type Actions = ActionsUnion<typeof Actions>;
