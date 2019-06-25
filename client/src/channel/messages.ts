import { ActionCreatorsMapObject } from "redux";
import { FollowupPart, MsgPartInitialDataset, MsgPartJob } from "../messages";

export interface Message<T extends string> {
    messageType: T
}

export function createMessage<T extends string, O>(messageType: T, attribs: O) {
    return Object.assign({ messageType }, attribs);
}

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
    CANCEL_JOB_DONE = "CANCEL_JOB_DONE",
}

export const Messages = {
    initialState: (jobs: MsgPartJob[], datasets: MsgPartInitialDataset[]) => createMessage(MessageTypes.INITIAL_STATE, { jobs, datasets }),
    startJob: (job: string, dataset: string) => createMessage(MessageTypes.JOB_STARTED, { job, dataset }),
    finishJob: (job: string, followup: FollowupPart) => createMessage(MessageTypes.FINISH_JOB, { job, followup }),
    taskResult: (job: string, followup: FollowupPart) => createMessage(MessageTypes.TASK_RESULT, { job, followup }),
    jobError: (job: string, msg: string) => createMessage(MessageTypes.JOB_ERROR, { job, msg }),
    binary: (objectURL: string) => createMessage(MessageTypes.BINARY, { objectURL }),
    open: () => createMessage(MessageTypes.OPEN, {}),
    close: () => createMessage(MessageTypes.CLOSE, {}),
    error: (msg: string) => createMessage(MessageTypes.ERROR, { msg }),
    deleteDataset: (dataset: string) => createMessage(MessageTypes.DELETE_DATASET, { dataset }),
    cancelled: (job: string) => createMessage(MessageTypes.CANCEL_JOB_DONE, { job }),
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