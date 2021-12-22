import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisDetails, AnalysisTypes, CompoundAnalysisDetails } from "../messages";
import { CompoundAnalysis } from "./types";


export enum ActionTypes {
    CREATE = 'COMPOUND_ANALYSIS_CREATE',
    CREATED = 'COMPOUND_ANALYSIS_CREATED',
    SET_PARAMS = 'COMPOUND_ANALYSIS_SET_PARAMS',
    UPDATED = 'COMPOUND_ANALYSIS_UPDATED',
    RUN = 'COMPOUND_ANALYSIS_RUN',
    RUNNING = 'COMPOUND_ANALYSIS_RUNNING',
    REMOVE = 'COMPOUND_ANALYSIS_REMOVE',
    REMOVED = 'COMPOUND_ANALYSIS_REMOVED',
    ERROR = 'COMPOUND_ANALYSIS_ERROR',
    ENABLE_AUTOSTART = 'COMPOUND_ANALYSIS_ENABLE_AUTOSTART',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (
        compoundAnalysis: CompoundAnalysis, autoStart: boolean
    ) => createAction(ActionTypes.CREATED, { compoundAnalysis, autoStart }),
    setParams: (
        compoundAnalysis: CompoundAnalysis,
        analysisIndex: number, details: AnalysisDetails, analysisId?: string,
    ) => createAction(ActionTypes.SET_PARAMS, { compoundAnalysis, analysisId, analysisIndex, details }),
    enableAutoStart: (
        compoundAnalysisId: string
    ) => createAction(ActionTypes.ENABLE_AUTOSTART, { compoundAnalysisId }),
    updated: (id: string, details: CompoundAnalysisDetails) => createAction(ActionTypes.UPDATED, { id, details }),
    run: (id: string, analysisIndex: number, details: AnalysisDetails) => createAction(ActionTypes.RUN, { id, analysisIndex, details }),
    running: (id: string, job: string, analysisIndex: number) => createAction(ActionTypes.RUNNING, { id, job, jobIndex: analysisIndex }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    removed: (id: string) => createAction(ActionTypes.REMOVED, { id }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;

export type ActionParts = {
    [K in keyof typeof Actions]: ReturnType<typeof Actions[K]>
}
