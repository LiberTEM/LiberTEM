import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, filterWithPred, insertById, updateById } from "../helpers/reducerHelpers";
import * as analysisActions from "./actions";
import { AnalysisState } from "./types";

export type AnalysisReducerState = ById<AnalysisState>;

const initialAnalysisState: AnalysisReducerState = {
    byId: {},
    ids: [],
}

export function analysisReducer(state = initialAnalysisState, action: AllActions) {
    switch (action.type) {
        case analysisActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.analysis.id, action.payload.analysis);
        }
        case analysisActions.ActionTypes.UPDATE_PARAMETERS: {
            const details = state.byId[action.payload.id].details;
            const newDetails = Object.assign({}, details, {
                parameters: Object.assign({}, details.parameters, action.payload.parameters),
            })
            return updateById(state, action.payload.id, {
                details: newDetails,
            });
        }
        case analysisActions.ActionTypes.RUN: {
            return updateById(state, action.payload.id, { status: "busy" });
        }
        case channelActions.ActionTypes.FINISH_JOB: {
            const idleIfJobMatch = ((part: AnalysisState) => {
                if (action.payload.job === part.currentJob) {
                    return Object.assign({}, part, { status: "idle" });
                } else {
                    return part;
                }
            })
            const byId = state.ids.reduce((acc, id) => {
                return Object.assign({}, acc, {
                    [id]: idleIfJobMatch(state.byId[id]),
                });
            }, {});
            return {
                byId,
                ids: state.ids,
            }
        }
        case analysisActions.ActionTypes.RUNNING: {
            return updateById(state, action.payload.id, { currentJob: action.payload.job })
        }
        case analysisActions.ActionTypes.REMOVE: {
            return filterWithPred(state, (r: AnalysisState) => r.id !== action.payload.id);
        }
        case analysisActions.ActionTypes.SET_PREVIEW: {
            return updateById(state, action.payload.id, {
                preview: action.payload.preview,
            })
        }
        case analysisActions.ActionTypes.SET_PREVIEW_MODE: {
            const newPreview = Object.assign({}, state.byId[action.payload.id].preview, {
                mode: action.payload.mode,
            });
            return updateById(state, action.payload.id, { preview: newPreview });
        }
    }
    return state;
}