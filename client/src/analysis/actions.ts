import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisDetails, AnalysisTypes } from "../messages";
import { AnalysisState } from "./types";


export enum ActionTypes {
    CREATE = 'ANALYSIS_CREATE',
    CREATED = 'ANALYSIS_CREATED',
    PREPARE_RUN = 'ANALYSIS_PREPARE_RUN',
    RUN = 'ANALYSIS_RUN',
    RUNNING = 'ANALYSIS_RUNNING',
    REMOVE = 'ANALYSIS_REMOVE',
    REMOVED = 'ANALYSIS_REMOVED',
    ERROR = 'ANALYSIS_ERROR',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (analysis: AnalysisState) => createAction(ActionTypes.CREATED, { analysis }),
    prepareRun: (id: string, jobIndex: number, job: string) => createAction(ActionTypes.PREPARE_RUN, { id, jobIndex, job }),
    run: (id: string, jobIndex: number, parameters: AnalysisDetails) => createAction(ActionTypes.RUN, { id, jobIndex, parameters }),
    running: (id: string, job: string, jobIndex: number) => createAction(ActionTypes.RUNNING, { id, job, jobIndex }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    removed: (id: string) => createAction(ActionTypes.REMOVED, { id }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;

export type ActionParts = {
    [K in keyof typeof Actions]: ReturnType<typeof Actions[K]>
}