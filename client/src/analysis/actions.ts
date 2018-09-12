import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisParameters, AnalysisTypes } from "../messages";
import { AnalysisState, FramePreview, PreviewMode } from "./types";


export enum ActionTypes {
    CREATE = 'ANALYSIS_CREATE',
    CREATED = 'ANALYSIS_CREATED',
    UPDATE_PARAMETERS = 'ANALYSIS_UPDATE_PARAMETERS',
    RUN = 'ANALYSIS_RUN',
    RUNNING = 'ANALYSIS_RUNNING',
    REMOVE = 'ANALYSIS_REMOVE',
    REMOVED = 'ANALYSIS_REMOVED',
    ERROR = 'ANALYSIS_ERROR',
    SET_PREVIEW = 'ANALYSIS_SET_PREVIEW',
    SET_PREVIEW_MODE = 'ANALYSIS_SET_PREVIEW_MODE',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (analysis: AnalysisState) => createAction(ActionTypes.CREATED, { analysis }),
    updateParameters: (id: string, parameters: Partial<AnalysisParameters>) => createAction(ActionTypes.UPDATE_PARAMETERS, { id, parameters }),
    setPreview: (id: string, preview: FramePreview) => createAction(ActionTypes.SET_PREVIEW, { id, preview }),
    setPreviewMode: (id: string, mode: PreviewMode) => createAction(ActionTypes.SET_PREVIEW_MODE, { id, mode }),
    run: (id: string) => createAction(ActionTypes.RUN, { id }),
    running: (id: string, job: string) => createAction(ActionTypes.RUNNING, { id, job }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    removed: (id: string) => createAction(ActionTypes.REMOVED, { id }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;