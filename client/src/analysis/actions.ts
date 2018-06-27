import { ActionsUnion, createAction } from "../helpers/actionHelpers";
import { Analysis, AnalysisParameters, AnalysisTypes } from "./types";


export enum ActionTypes {
    CREATE = 'ANALYSIS_CREATE',
    CREATED = 'ANALYSIS_CREATED',
    UPDATE_PARAMETERS = 'ANALYSIS_UPDATE_PARAMETERS',
    RUN = 'ANALYSIS_RUN',
    RUNNING = 'ANALYSIS_RUNNING',
    REMOVE = 'ANALYSIS_REMOVE',
}

export const Actions = {
    create: (dataset: string, analysisType: AnalysisTypes) => createAction(ActionTypes.CREATE, { dataset, analysisType }),
    created: (analysis: Analysis) => createAction(ActionTypes.CREATED, { analysis }),
    updateParameters: (id: string, parameters: AnalysisParameters) => createAction(ActionTypes.UPDATE_PARAMETERS, { id, parameters }),
    run: (id: string) => createAction(ActionTypes.RUN, { id }),
    running: (id: string, job: string) => createAction(ActionTypes.RUNNING, { id, job }),
    remove: (id: string) => createAction(ActionTypes.REMOVE, { id }),
}

export type Actions = ActionsUnion<typeof Actions>;