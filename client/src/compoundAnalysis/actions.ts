import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisDetails, AnalysisTypes } from "../messages";
import { CompoundAnalysisState } from "./types";


export enum ActionTypes {
    CREATE = 'COMPOUND_ANALYSIS_CREATE',
    CREATED = 'COMPOUND_ANALYSIS_CREATED',
    PREPARE_RUN = 'COMPOUND_ANALYSIS_PREPARE_RUN',
    RUN = 'COMPOUND_ANALYSIS_RUN',
    RUNNING = 'COMPOUND_ANALYSIS_RUNNING',
    REMOVE = 'COMPOUND_ANALYSIS_REMOVE',
    REMOVED = 'COMPOUND_ANALYSIS_REMOVED',
    ERROR = 'COMPOUND_ANALYSIS_ERROR',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (analysis: CompoundAnalysisState) => createAction(ActionTypes.CREATED, { analysis }),
    prepareRun: (id: string, analysisIndex: number, job: string) => createAction(ActionTypes.PREPARE_RUN, { id, analysisIndex, job }),
    run: (id: string, analysisIndex: number, parameters: AnalysisDetails) => createAction(ActionTypes.RUN, { id, analysisIndex, parameters }),
    running: (id: string, job: string, analysisIndex: number) => createAction(ActionTypes.RUNNING, { id, job, jobIndex: analysisIndex }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    removed: (id: string) => createAction(ActionTypes.REMOVED, { id }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;

export type ActionParts = {
    [K in keyof typeof Actions]: ReturnType<typeof Actions[K]>
}