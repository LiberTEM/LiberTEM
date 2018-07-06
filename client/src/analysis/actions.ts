import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { AnalysisParameters, AnalysisTypes } from "../messages";
import { AnalysisState } from "./types";


export enum ActionTypes {
    CREATE = 'ANALYSIS_CREATE',
    CREATED = 'ANALYSIS_CREATED',
    UPDATE_PARAMETERS = 'ANALYSIS_UPDATE_PARAMETERS',
    RUN = 'ANALYSIS_RUN',
    RUNNING = 'ANALYSIS_RUNNING',
    REMOVE = 'ANALYSIS_REMOVE',
    ERROR = 'ANALYSIS_ERROR',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (analysis: AnalysisState) => createAction(ActionTypes.CREATED, { analysis }),
    updateParameters: (id: string, parameters: AnalysisParameters) => createAction(ActionTypes.UPDATE_PARAMETERS, { id, parameters }),
    run: (id: string) => createAction(ActionTypes.RUN, { id }),
    running: (id: string, job: string) => createAction(ActionTypes.RUNNING, { id, job }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
    error: (msg: string, timestamp: number) => createAction(ActionTypes.ERROR, { msg, timestamp }),
}

export type Actions = ActionsUnion<typeof Actions>;