import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisDetails, AnalysisTypes } from "../messages";
import { AnalysisState } from "./types";


export enum ActionTypes {
    CREATE = 'ANALYSIS_CREATE',
    CREATED = 'ANALYSIS_CREATED',
    UPDATED = 'ANALYSIS_UPDATED',
    REMOVE = 'ANALYSIS_REMOVE',
    REMOVED = 'ANALYSIS_REMOVED',
    ERROR = 'ANALYSIS_ERROR',
}

export const Actions = {
    create: (
        dataset: string, analysisType: AnalysisTypes
    ) => createAction(ActionTypes.CREATE, { dataset, analysisType }),

    created: (
        analysis: AnalysisState, compoundAnalysis: string, analysisIndex: number
    ) => createAction(ActionTypes.CREATED, {
        analysis, compoundAnalysis, analysisIndex
    }),

    updated: (
        id: string, details: AnalysisDetails
    ) => createAction(ActionTypes.UPDATED, { id, details }),

    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    removed: (id: string) => createAction(ActionTypes.REMOVED, { id }),
    error: (
        msg: string, timestamp: number, id: string
    ) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;

export type ActionParts = {
    [K in keyof typeof Actions]: ReturnType<typeof Actions[K]>
}
