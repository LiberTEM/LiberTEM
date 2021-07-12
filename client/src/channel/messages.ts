import { ActionCreatorsMapObject } from "redux";
import { AnalysisDetails, CompoundAnalysisDetails, CreateDatasetMessage, FollowupPart, MsgPartAnalysis, MsgPartCompoundAnalysis, MsgPartInitialDataset, MsgPartJob } from "../messages";

export interface Message<T extends string> {
    messageType: T
}

export const createMessage = <T extends string, O>(messageType: T, attribs: O) => Object.assign({ messageType }, attribs);

export enum MessageTypes {
    INITIAL_STATE = "INITIAL_STATE",
    JOB_STARTED = "JOB_STARTED",
    FINISH_JOB = "FINISH_JOB",
    TASK_RESULT = "TASK_RESULT",
    JOB_ERROR = "JOB_ERROR",
    BINARY = "BINARY",
    OPEN = "OPEN",
    CLOSE = "CLOSE",
    ERROR = "ERROR",
    DELETE_DATASET = "DELETE_DATASET",
    CREATE_DATASET = "CREATE_DATASET",
    CANCEL_JOB_DONE = "CANCEL_JOB_DONE",
    CANCEL_JOB_FAILED = "CANCEL_JOB_FAILED",
    ANALYSIS_CREATED = "ANALYSIS_CREATED",
    ANALYSIS_UPDATED = "ANALYSIS_UPDATED",
    ANALYSIS_REMOVED = "ANALYSIS_REMOVED",
    COMPOUND_ANALYSIS_CREATED = "COMPOUND_ANALYSIS_CREATED",
    COMPOUND_ANALYSIS_UPDATED = "COMPOUND_ANALYSIS_UPDATED",
    COMPOUND_ANALYSIS_REMOVED = "COMPOUND_ANALYSIS_REMOVED",
}

export const Messages = {
    initialState: (
        jobs: MsgPartJob[],
        datasets: MsgPartInitialDataset[],
        analyses: MsgPartAnalysis[],
        compoundAnalyses: MsgPartCompoundAnalysis[]
    ) => createMessage(MessageTypes.INITIAL_STATE, {
        jobs, datasets, compoundAnalyses, analyses,
    }),

    startJob: (job: string) => createMessage(MessageTypes.JOB_STARTED, { job }),
    finishJob: (job: string, followup: FollowupPart) => createMessage(MessageTypes.FINISH_JOB, { job, followup }),
    taskResult: (job: string, followup: FollowupPart) => createMessage(MessageTypes.TASK_RESULT, { job, followup }),
    jobError: (job: string, msg: string) => createMessage(MessageTypes.JOB_ERROR, { job, msg }),
    binary: (objectURL: string) => createMessage(MessageTypes.BINARY, { objectURL }),
    open: () => createMessage(MessageTypes.OPEN, {}),
    close: () => createMessage(MessageTypes.CLOSE, {}),
    error: (msg: string) => createMessage(MessageTypes.ERROR, { msg }),
    deleteDataset: (dataset: string) => createMessage(MessageTypes.DELETE_DATASET, { dataset }),
    createDataset: (dataset: string, details: CreateDatasetMessage) => createMessage(MessageTypes.CREATE_DATASET, { dataset, details }),
    cancelled: (job: string) => createMessage(MessageTypes.CANCEL_JOB_DONE, { job }),
    cancelFailed: (job: string) => createMessage(MessageTypes.CANCEL_JOB_FAILED, { job }),
    analysisCreated: (analysis: string, dataset: string, details: AnalysisDetails) => createMessage(MessageTypes.ANALYSIS_CREATED, { dataset, analysis, details }),
    analysisUpdated: (analysis: string, dataset: string, details: AnalysisDetails) => createMessage(MessageTypes.ANALYSIS_UPDATED, { dataset, analysis, details }),
    analysisRemoved: (analysis: string) => createMessage(MessageTypes.ANALYSIS_REMOVED, { analysis }),

    compoundAnalysisCreated: (compoundAnalysis: string, dataset: string, details: CompoundAnalysisDetails) => createMessage(MessageTypes.COMPOUND_ANALYSIS_CREATED, { dataset, compoundAnalysis, details }),
    compoundAnalysisUpdated: (compoundAnalysis: string, dataset: string, details: CompoundAnalysisDetails) => createMessage(MessageTypes.COMPOUND_ANALYSIS_UPDATED, { dataset, compoundAnalysis, details }),
    compoundAnalysisRemoved: (compoundAnalysis: string) => createMessage(MessageTypes.ANALYSIS_REMOVED, { compoundAnalysis }),
}

export type MessagesUnion<A extends ActionCreatorsMapObject> = ReturnType<A[keyof A]>
export type Messages = MessagesUnion<typeof Messages>;

// types of messages sent by the server:
/*
export type InitialStateMessage = ReturnType<typeof Messages.initialState>;
export type StartJobMessage = ReturnType<typeof Messages.startJob>;
export type FinishJobMessage = ReturnType<typeof Messages.finishJob>;
export type TaskResultMessage = ReturnType<typeof Messages.taskResult>;
*/

export type BinaryMessage = ReturnType<typeof Messages.binary>;