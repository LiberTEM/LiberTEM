import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisParameters, AnalysisTypes } from "../messages";
import { AnalysisState, FrameMode, JobKind } from "./types";


export enum ActionTypes {
    CREATE = 'ANALYSIS_CREATE',
    CREATED = 'ANALYSIS_CREATED',
    UPDATE_PARAMETERS = 'ANALYSIS_UPDATE_PARAMETERS',
    RUN = 'ANALYSIS_RUN',
    RUNNING = 'ANALYSIS_RUNNING',
    REMOVE = 'ANALYSIS_REMOVE',
    REMOVED = 'ANALYSIS_REMOVED',
    ERROR = 'ANALYSIS_ERROR',
    SET_FRAMEVIEW_MODE = 'ANALYSIS_SET_FRAMEVIEW_MODE',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (analysis: AnalysisState) => createAction(ActionTypes.CREATED, { analysis }),
    updateParameters: (id: string, parameters: Partial<AnalysisParameters>, kind: JobKind) => createAction(ActionTypes.UPDATE_PARAMETERS, { id, kind, parameters }),
    setFrameViewMode: (id: string, mode: FrameMode, initialParams: Partial<AnalysisParameters>) => createAction(ActionTypes.SET_FRAMEVIEW_MODE, { id, mode, initialParams }),
    run: (id: string, kind: JobKind) => createAction(ActionTypes.RUN, { id, kind }),
    running: (id: string, job: string, kind: JobKind) => createAction(ActionTypes.RUNNING, { id, job, kind }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    removed: (id: string) => createAction(ActionTypes.REMOVED, { id }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;

export type ActionParts = {
    [K in keyof typeof Actions]: ReturnType<typeof Actions[K]>
}