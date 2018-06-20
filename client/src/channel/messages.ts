import { ActionCreatorsMapObject } from "redux";

export interface Message<T extends string> {
    messageType: T
}

export function createMessage<T extends string, O>(messageType: T, attribs: O) {
    return Object.assign({ messageType }, attribs);
}

export enum MessageTypes {
    INITIAL_STATE = "INITIAL_STATE",
    START_JOB = "START_JOB",
    FINISH_JOB = "FINISH_JOB",
    TASK_RESULT = "TASK_RESULT",
    BINARY = "BINARY",
    OPEN = "OPEN",
    CLOSE = "CLOSE",
    ERROR = "ERROR",
}

export interface FollowupPart {
    numMessages: number,
}

export interface MsgPartJob {
    job: string,
    dataset: string,
}

export interface MsgPartDataset {
    dataset: string,
    name: string,
    path: string,
    tileshape: number[],
    type: string,
}

// tslint:disable:object-literal-sort-keys
export const Messages = {
    initialState: (jobs: MsgPartJob[], datasets: MsgPartDataset[]) => createMessage(MessageTypes.INITIAL_STATE, { jobs, datasets }),
    startJob: (job: string) => createMessage(MessageTypes.START_JOB, { job }),
    finishJob: (job: string, followup: FollowupPart) => createMessage(MessageTypes.FINISH_JOB, { job, followup }),
    taskResult: (job: string, followup: FollowupPart) => createMessage(MessageTypes.TASK_RESULT, { job, followup }),
    binary: (objectURL: string) => createMessage(MessageTypes.BINARY, { objectURL }),
    open: () => createMessage(MessageTypes.OPEN, {}),
    close: () => createMessage(MessageTypes.CLOSE, {}),
    error: (msg: string) => createMessage(MessageTypes.ERROR, { msg }),
}
// tslint:enable

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